// DatcomWindForce: apply the physical directional-drag wind force to a link every PreUpdate step.
//   F_body_i = 0.5 * rho * CdS_i * |Vrel_body_i| * Vrel_body_i   (signed quadratic, per body axis)
//   Vrel = V_wind(world) - V_body(world), rotated into the body frame; force rotated back to world.
// Applied ONCE per step via Link::AddWorldForce (NOT the accumulating ApplyLinkWrench topic — see
// memory gz-applylinkwrench-accumulates). Wind vector (world frame, m/s) comes from a gz topic
// (default /datcom/wind, gz.msgs.Vector3d), settable at any rate; the plugin only reads the latest.
#include <mutex>
#include <string>
#include <cmath>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/vector3d.pb.h>
#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>
#include <gz/common/Console.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>

namespace datcom_wind
{
class DatcomWindForce
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
  public: void Configure(const gz::sim::Entity &_entity,
              const std::shared_ptr<const sdf::Element> &_sdf,
              gz::sim::EntityComponentManager &,
              gz::sim::EventManager &) override
  {
    this->selfEntity = _entity;   // may be a world (world-attached) or a model (model-attached)
    if (_sdf->HasElement("model_name")) this->modelName = _sdf->Get<std::string>("model_name");
    if (_sdf->HasElement("link_name")) this->linkName = _sdf->Get<std::string>("link_name");
    if (_sdf->HasElement("rho"))   this->rho    = _sdf->Get<double>("rho");
    if (_sdf->HasElement("cds_x")) this->cds[0] = _sdf->Get<double>("cds_x");
    if (_sdf->HasElement("cds_y")) this->cds[1] = _sdf->Get<double>("cds_y");
    if (_sdf->HasElement("cds_z")) this->cds[2] = _sdf->Get<double>("cds_z");
    // Optional fixed-wing aero (teTra Mk-7 LS(1)-0417 wing @0deg incidence). Default 0 -> disabled (parasite only).
    //   lift L = 0.5*rho*CL0*S*Vh^2 along body +z (up), from the horizontal relative wind Vh; this offloads the
    //   rotors in a steady wind. induced drag Fx = 0.5*rho*CdS_i*|Vrel_x|*Vrel_x adds to the fore-aft parasite.
    if (_sdf->HasElement("cl0"))    this->cl0    = _sdf->Get<double>("cl0");
    if (_sdf->HasElement("wing_s")) this->wingS  = _sdf->Get<double>("wing_s");
    if (_sdf->HasElement("cds_i"))  this->cdsI   = _sdf->Get<double>("cds_i");
    std::string topic = _sdf->HasElement("wind_topic") ? _sdf->Get<std::string>("wind_topic") : std::string("/datcom/wind");
    this->node.Subscribe(topic, &DatcomWindForce::OnWind, this);
    gzmsg << "[DatcomWindForce] model=" << this->modelName << " link=" << this->linkName << " rho=" << this->rho
          << " CdS=(" << this->cds[0] << "," << this->cds[1] << "," << this->cds[2]
          << ") m^2 topic=" << topic << " (lazy link resolve)\n";
  }

  // Resolve base_link lazily: the model may be spawned AFTER world load (PX4_GZ_MODEL). Try self as a model
  // first; else find the model by name in the world.
  private: void ResolveLink(gz::sim::EntityComponentManager &_ecm)
  {
    gz::sim::Entity me = gz::sim::kNullEntity;
    { gz::sim::Model m(this->selfEntity); if (m.Valid(_ecm)) me = this->selfEntity; }
    if (me == gz::sim::kNullEntity)
      me = _ecm.EntityByComponents(gz::sim::components::Name(this->modelName), gz::sim::components::Model());
    if (me == gz::sim::kNullEntity) return;
    gz::sim::Model model(me);
    this->linkEntity = model.LinkByName(_ecm, this->linkName);
    if (this->linkEntity != gz::sim::kNullEntity)
    {
      gz::sim::Link(this->linkEntity).EnableVelocityChecks(_ecm, true);
      gzmsg << "[DatcomWindForce] resolved link '" << this->linkName << "' of model '" << this->modelName << "'\n";
    }
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
              gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused) return;
    if (this->linkEntity == gz::sim::kNullEntity) { this->ResolveLink(_ecm); if (this->linkEntity == gz::sim::kNullEntity) return; }
    gz::sim::Link link(this->linkEntity);
    auto pose = link.WorldPose(_ecm);
    auto vel  = link.WorldLinearVelocity(_ecm);
    if (!pose.has_value()) return;
    gz::math::Vector3d vbody = vel.has_value() ? vel.value() : gz::math::Vector3d::Zero;
    gz::math::Vector3d w;
    { std::lock_guard<std::mutex> lk(this->mtx); w = this->wind; }
    gz::math::Vector3d vrelWorld = w - vbody;                       // relative wind, world frame
    gz::math::Vector3d vrelBody  = pose->Rot().Inverse() * vrelWorld;
    gz::math::Vector3d fBody;
    for (int i = 0; i < 3; ++i)
    {
      double v = vrelBody[i];
      fBody[i] = 0.5 * this->rho * this->cds[i] * std::fabs(v) * v;  // signed quadratic drag per body axis
    }
    if (this->cl0 > 0.0)  // fixed-wing lift + induced drag from the horizontal relative wind
    {
      const double vh2 = vrelBody[0] * vrelBody[0] + vrelBody[1] * vrelBody[1];  // horizontal rel-wind^2
      fBody[2] += 0.5 * this->rho * this->cl0 * this->wingS * vh2;               // wing lift, body +z (up)
      const double vx = vrelBody[0];
      fBody[0] += 0.5 * this->rho * this->cdsI * std::fabs(vx) * vx;             // induced drag, along fore-aft
    }
    gz::math::Vector3d fWorld = pose->Rot() * fBody;
    link.AddWorldForce(_ecm, fWorld);                               // applied once this step (no accumulation)
  }

  private: void OnWind(const gz::msgs::Vector3d &_m)
  {
    std::lock_guard<std::mutex> lk(this->mtx);
    this->wind.Set(_m.x(), _m.y(), _m.z());
  }

  private: gz::sim::Entity selfEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity linkEntity{gz::sim::kNullEntity};
  private: std::string modelName{"teTra_mk-7_EM2_0"};
  private: std::string linkName{"base_link"};
  private: double rho{1.188};
  private: double cds[3]{0.46, 5.79, 23.5};
  private: double cl0{0.0};    // wing CL0 (0 = wing aero disabled)
  private: double wingS{0.0};  // wing reference area [m^2]
  private: double cdsI{0.0};   // induced drag area CdS_i [m^2]
  private: gz::math::Vector3d wind{0, 0, 0};
  private: std::mutex mtx;
  private: gz::transport::Node node;
};
}

GZ_ADD_PLUGIN(datcom_wind::DatcomWindForce, gz::sim::System,
    datcom_wind::DatcomWindForce::ISystemConfigure,
    datcom_wind::DatcomWindForce::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(datcom_wind::DatcomWindForce, "datcom_wind::DatcomWindForce")

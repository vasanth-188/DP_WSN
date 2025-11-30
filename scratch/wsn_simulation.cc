/* wsn_dqn.cc
   Simple NS-3 WSN skeleton that sends state to a Python bridge over UDP
   and applies binary wake/sleep decisions returned by Python.
   Usage: ./ns3 run scratch/wsn_dqn.cc
*/

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/energy-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/netanim-module.h"
#include <iostream>
#include <fstream>  // for file output
#include <sstream>
#include <iomanip>  // for setprecision
#include <string>
#include <vector>
#include <random>

#include <nlohmann/json.hpp> // JSON helper - see build notes

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("WSN-DQN");

static const std::string PY_BRIDGE_ADDR = "127.0.0.1";
static const uint16_t PY_BRIDGE_PORT = 5000;
static const uint16_t NS3_RECV_PORT = 5001;

struct NodeState {
    Ptr<Node> node;
    bool awake;
    double energy; // residual energy (J)
    uint32_t txPackets;
    uint32_t rxPackets;
};

class WsnDqnSim {
public:
    WsnDqnSim();
    void Run();

private:
    void CreateNodes();
    void SetupMobility();
    void InstallInternet();
    void SetupTraffic();
    void SetupUdpSocketToPython();
    void SendStateToPython();
    void HandlePythonResponse(Ptr<Socket> socket);
    void ApplyAction(const json &jaction);
    void EpochEvent();
    void UpdateNodeVisuals();  // Update NetAnim node colors based on state
    void ControlNodeTraffic(); // Start/stop traffic based on awake state
    void SaveStateHistory();   // Save simulation history to JSON file

    NodeContainer nodes;
    std::vector<NodeState> nstates;
    json stateHistory;         // Store all epochs for file output
    std::vector<ApplicationContainer> sensorApps;  // Per-node apps for control
    Ptr<Socket> txSocket; // socket to python
    Ptr<Socket> rxSocket; // socket to receive actions
    double epochInterval;
    uint32_t numNodes;
    uint32_t packetSize;
    EventId epochEvent;
    CsmaHelper csma;
    NetDeviceContainer devices;
    Ipv4InterfaceContainer interfaces;
    AnimationInterface *anim;
    uint32_t sinkRxCount;  // Track sink received packets
};

WsnDqnSim::WsnDqnSim()
: epochInterval(1.0), numNodes(25), packetSize(64), sinkRxCount(0)
{}

void WsnDqnSim::CreateNodes() {
    nodes.Create(numNodes);
    nstates.resize(numNodes);
    for (uint32_t i=0;i<numNodes;i++) {
        nstates[i].node = nodes.Get(i);
        nstates[i].awake = true;
        nstates[i].energy = 100.0; // arbitrary energy units
        nstates[i].txPackets = 0;
        nstates[i].rxPackets = 0;
    }
}

void WsnDqnSim::SetupMobility() {
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
    // random uniform 100x100 area
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    for (uint32_t i=0;i<numNodes;i++) {
        double x = dist(rng);
        double y = dist(rng);
        pos->Add(Vector(x,y,0.0));
    }
    mobility.SetPositionAllocator(pos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
}

void WsnDqnSim::InstallInternet() {
    InternetStackHelper internet;
    internet.Install(nodes);

    csma.SetChannelAttribute("DataRate", StringValue("10Mbps"));
    csma.SetChannelAttribute("Delay", StringValue("2ms"));
    devices = csma.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");
    interfaces = ipv4.Assign(devices);
}

void WsnDqnSim::SetupTraffic() {
    // Install real ns-3 UDP traffic from sensor nodes to sink (node 0)
    // Traffic is controlled by awake state - sleeping nodes don't transmit
    
    uint16_t sinkPort = 9;
    
    // Install PacketSink on node 0 to receive traffic
    PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                 InetSocketAddress(Ipv4Address::GetAny(), sinkPort));
    ApplicationContainer sinkApp = sinkHelper.Install(nodes.Get(0));
    sinkApp.Start(Seconds(0.1));
    sinkApp.Stop(Seconds(59.9));
    
    // Reserve space for per-node applications
    sensorApps.resize(numNodes);
    
    // Install OnOffApplication on each sensor node (1..N-1) to send to sink
    // Initially all nodes are awake
    for (uint32_t i = 1; i < numNodes; i++) {
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(interfaces.GetAddress(0), sinkPort));
        onoff.SetAttribute("DataRate", StringValue("2kbps"));  // Higher rate to see effect
        onoff.SetAttribute("PacketSize", UintegerValue(64));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
        
        sensorApps[i] = onoff.Install(nodes.Get(i));
        // Stagger start times
        double startTime = 0.5 + (i * 0.05);
        sensorApps[i].Start(Seconds(startTime));
        sensorApps[i].Stop(Seconds(59.5));
    }
}

void WsnDqnSim::SetupUdpSocketToPython() {
    // TX socket: send state to Python (UDP)
    txSocket = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
    InetSocketAddress remote = InetSocketAddress(Ipv4Address(PY_BRIDGE_ADDR.c_str()), PY_BRIDGE_PORT);
    txSocket->Connect(remote);

    // RX socket: bind to receive port for actions from Python
    // use node 0 (sink) to receive Python replies so socket factory exists
    rxSocket = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), NS3_RECV_PORT);
    rxSocket->Bind(local);
    rxSocket->SetRecvCallback(MakeCallback(&WsnDqnSim::HandlePythonResponse, this));
}

void WsnDqnSim::SendStateToPython() {
    // Prepare a JSON object with per-node state (awake, energy, tx, rx)
    json j;
    j["num_nodes"] = numNodes;
    j["epoch"] = Simulator::Now().GetSeconds();
    j["nodes"] = json::array();
    for (uint32_t i=0;i<numNodes;i++) {
        json nj;
        nj["id"] = (int)i;
        nj["awake"] = nstates[i].awake;
        nj["energy"] = nstates[i].energy;
        nj["tx"] = nstates[i].txPackets;
        nj["rx"] = nstates[i].rxPackets;
        j["nodes"].push_back(nj);
    }
    std::string s = j.dump();
    Ptr<Packet> p = Create<Packet>((uint8_t*)s.c_str(), s.size());
    // send as UDP datagram
    txSocket->Send(p);
    NS_LOG_INFO("Sent state to Python: " << s);
}

void WsnDqnSim::HandlePythonResponse(Ptr<Socket> socket) {
    Address from;
    Ptr<Packet> p = socket->RecvFrom(from);
    uint32_t sz = p->GetSize();
    std::vector<uint8_t> buf(sz);
    p->CopyData(buf.data(), sz);
    std::string s((char*)buf.data(), sz);
    NS_LOG_INFO("Received action: " << s);
    try {
        json j = json::parse(s);
        ApplyAction(j);
    } catch (...) {
        NS_LOG_WARN("Failed to parse python response");
    }
}

void WsnDqnSim::ApplyAction(const json &jaction) {
    // Expect jaction["wake_list"] = [0,0,1,1,...] length numNodes (1->wake,0->sleep)
    if (!jaction.contains("wake_list")) {
        NS_LOG_WARN("No wake_list in action");
        return;
    }
    auto arr = jaction["wake_list"];
    for (uint32_t i=0;i<std::min((size_t)numNodes, arr.size()); i++) {
        bool wake = (int)arr[i] != 0;
        bool wasAwake = nstates[i].awake;
        nstates[i].awake = wake;
        
        // Control traffic based on wake state (skip sink node 0)
        if (i > 0 && i < sensorApps.size() && sensorApps[i].GetN() > 0) {
            Ptr<Application> app = sensorApps[i].Get(0);
            if (wake && !wasAwake) {
                // Waking up - resume transmission
                // OnOff app will transmit when running
            } else if (!wake && wasAwake) {
                // Going to sleep - stop transmission
                // Node sleeps = no energy drain from TX, but also no packets
            }
        }
    }
    
    // Log metrics
    uint32_t awakeCount = 0;
    for (auto &ns : nstates) if (ns.awake) awakeCount++;
    NS_LOG_INFO("Applied action. Awake nodes: " << awakeCount << "/" << (numNodes-1));
}

void WsnDqnSim::UpdateNodeVisuals() {
    // Update node colors in NetAnim based on energy and awake state
    // COLOR SCHEME:
    //   Sink (node 0): Blue, large and constant
    //   AWAKE nodes: Green gradient based on energy (bright green -> yellow -> orange -> red)
    //   ASLEEP nodes: Purple/dark (clearly distinct from awake)
    //   Dead nodes (energy=0): Black, tiny
    // SIZE: Proportional to energy level (larger = more energy)
    
    for (uint32_t i = 0; i < numNodes; i++) {
        if (i == 0) {
            // Sink node: always blue, size 4 (largest)
            anim->UpdateNodeColor(i, 30, 144, 255);  // Dodger Blue
            anim->UpdateNodeSize(i, 4.0, 4.0);
            anim->UpdateNodeDescription(i, "⬟ SINK");
            continue;
        }
        
        double energyPct = nstates[i].energy / 100.0;  // 0.0 to 1.0
        
        // Size proportional to energy: min 0.5, max 2.5
        double nodeSize = 0.5 + 2.0 * energyPct;
        
        // Check if node is dead
        if (nstates[i].energy <= 0) {
            // Dead node: black, tiny
            anim->UpdateNodeColor(i, 0, 0, 0);
            anim->UpdateNodeSize(i, 0.3, 0.3);
            anim->UpdateNodeDescription(i, "✗ DEAD");
            continue;
        }
        
        if (!nstates[i].awake) {
            // ASLEEP: Purple/magenta tones (clearly different from awake green)
            // Darker purple for low energy, lighter for high energy
            uint8_t purple_r = (uint8_t)(128 + 60 * energyPct);  // 128-188
            uint8_t purple_b = (uint8_t)(180 + 75 * energyPct);  // 180-255
            anim->UpdateNodeColor(i, purple_r, 50, purple_b);    // Purple shades
            anim->UpdateNodeSize(i, nodeSize * 0.8, nodeSize * 0.8);  // Slightly smaller when asleep
            
            std::ostringstream desc;
            desc << "💤 N" << i << " [" << std::fixed << std::setprecision(0) << nstates[i].energy << "%]";
            anim->UpdateNodeDescription(i, desc.str());
        } else {
            // AWAKE: Green-Yellow-Orange-Red gradient based on energy
            uint8_t r, g, b;
            if (energyPct > 0.7) {
                // High energy: Bright green
                r = 0; g = 255; b = 50;
            } else if (energyPct > 0.5) {
                // Good energy: Yellow-green
                r = (uint8_t)(180 * (0.7 - energyPct) / 0.2);
                g = 255; b = 0;
            } else if (energyPct > 0.3) {
                // Medium energy: Yellow to orange
                r = 255;
                g = (uint8_t)(200 + 55 * (energyPct - 0.3) / 0.2);
                b = 0;
            } else if (energyPct > 0.1) {
                // Low energy: Orange
                r = 255; g = (uint8_t)(100 + 100 * (energyPct - 0.1) / 0.2); b = 0;
            } else {
                // Critical energy: Red
                r = 255; g = (uint8_t)(50 * energyPct / 0.1); b = 0;
            }
            
            anim->UpdateNodeColor(i, r, g, b);
            anim->UpdateNodeSize(i, nodeSize, nodeSize);
            
            std::ostringstream desc;
            desc << "⚡ N" << i << " [" << std::fixed << std::setprecision(0) << nstates[i].energy << "%]";
            anim->UpdateNodeDescription(i, desc.str());
        }
    }
}

void WsnDqnSim::ControlNodeTraffic() {
    // Start/stop OnOff applications based on awake state
    // Sleeping nodes don't transmit real packets
    for (uint32_t i = 1; i < numNodes; i++) {
        if (i >= sensorApps.size() || sensorApps[i].GetN() == 0) continue;
        
        Ptr<OnOffApplication> app = DynamicCast<OnOffApplication>(sensorApps[i].Get(0));
        if (!app) continue;
        
        if (nstates[i].awake) {
            // Node is awake - use normal data rate for transmission
            app->SetAttribute("DataRate", StringValue("2kbps"));
        } else {
            // Node is sleeping - set data rate to 0 to stop transmission
            app->SetAttribute("DataRate", StringValue("1bps"));  // Minimal rate when sleeping
        }
    }
}

void WsnDqnSim::EpochEvent() {
    // Called periodically; send state to Python, then schedule next epoch
    
    // Control traffic based on current awake states
    ControlNodeTraffic();
    
    // Send state to Python for DQN decision
    SendStateToPython();

    // Update energy and counters based on awake state
    for (uint32_t i=0;i<numNodes;i++) {
        if (i==0) continue; // node 0 is sink
        if (nstates[i].awake) {
            // Awake: transmitting, higher energy drain
            nstates[i].txPackets++;
            nstates[i].energy -= 3.0; // Higher drain for visible color changes
            // Track delivery (80% success rate when awake)
            bool delivered = (CreateObject<UniformRandomVariable>()->GetValue() > 0.2);
            if (delivered) {
                nstates[0].rxPackets++;
            }
        } else {
            // Sleeping: minimal energy drain, no transmission
            nstates[i].energy -= 0.3;
        }
        // Clamp energy to minimum 0
        if (nstates[i].energy < 0) nstates[i].energy = 0;
    }
    
    // Save state to history for file output
    json epochState;
    epochState["epoch"] = Simulator::Now().GetSeconds();
    epochState["nodes"] = json::array();
    uint32_t awakeCount = 0;
    double totalEnergy = 0;
    for (uint32_t i = 0; i < numNodes; i++) {
        json nj;
        nj["id"] = (int)i;
        nj["awake"] = nstates[i].awake;
        nj["energy"] = nstates[i].energy;
        nj["tx"] = nstates[i].txPackets;
        nj["rx"] = nstates[i].rxPackets;
        epochState["nodes"].push_back(nj);
        if (i > 0) {
            if (nstates[i].awake) awakeCount++;
            totalEnergy += nstates[i].energy;
        }
    }
    epochState["awake_count"] = awakeCount;
    epochState["avg_energy"] = totalEnergy / (numNodes - 1);
    epochState["sink_rx"] = nstates[0].rxPackets;
    stateHistory["epochs"].push_back(epochState);
    
    // Update node visuals based on current state
    UpdateNodeVisuals();
    
    // schedule next epoch
    Simulator::Schedule(Seconds(epochInterval), &WsnDqnSim::EpochEvent, this);
}

void WsnDqnSim::SaveStateHistory() {
    // Save all state history to JSON file for Python plotting
    stateHistory["num_nodes"] = numNodes;
    stateHistory["total_epochs"] = stateHistory["epochs"].size();
    
    std::ofstream outFile("wsn_dqn_history.json");
    if (outFile.is_open()) {
        outFile << stateHistory.dump(2);
        outFile.close();
        NS_LOG_INFO("Saved state history to wsn_dqn_history.json");
    } else {
        NS_LOG_WARN("Failed to save state history");
    }
}

void WsnDqnSim::Run() {
    CreateNodes();
    SetupMobility();
    InstallInternet();
    SetupTraffic();
    SetupUdpSocketToPython();
    
    // Initialize state history
    stateHistory["epochs"] = json::array();

    // Setup NetAnim animation with enhanced visuals
    anim = new AnimationInterface("wsn_dqn_anim.xml");
    anim->SetMobilityPollInterval(Seconds(0.1));
    
    // Enable packet tracing for NetAnim to show packet flows
    anim->EnablePacketMetadata(true);
    
    // Initial node setup: Sink is blue/large, sensors are green
    // Node 0 = SINK (blue, larger)
    anim->UpdateNodeDescription(0, "SINK");
    anim->UpdateNodeColor(0, 0, 100, 255);  // Blue
    anim->UpdateNodeSize(0, 3.0, 3.0);      // Larger size
    
    // Sensor nodes 1..N-1 (green, normal size)
    for (uint32_t i = 1; i < numNodes; i++) {
        std::ostringstream desc;
        desc << "N" << i << " (100%)";
        anim->UpdateNodeDescription(i, desc.str());
        anim->UpdateNodeColor(i, 0, 255, 0);  // Green = high energy
        anim->UpdateNodeSize(i, 2.0, 2.0);
    }

    // Schedule first epoch soon
    Simulator::Schedule(Seconds(0.5), &WsnDqnSim::EpochEvent, this);

    Simulator::Stop(Seconds(30.0));
    Simulator::Run();
    
    // Save simulation history to file for Python analysis
    SaveStateHistory();
    
    Simulator::Destroy();
}

int main(int argc, char *argv[]) {
    LogComponentEnable("WSN-DQN", LOG_LEVEL_INFO);
    WsnDqnSim sim;
    sim.Run();
    return 0;
}

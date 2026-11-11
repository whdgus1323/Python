//
// Copyright (C) 2014 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#include "inet/routing/aodv/Aodv.h"

#include <filesystem>
#include <fstream>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "inet/common/IProtocolRegistrationListener.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/stlutils.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/linklayer/ieee80211/mac/Ieee80211Mac.h"
#include "inet/networklayer/common/HopLimitTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/networklayer/common/L3Tools.h"
#include "inet/networklayer/ipv4/IcmpHeader.h"
#include "inet/networklayer/ipv4/Ipv4Header_m.h"
#include "inet/networklayer/ipv4/Ipv4Route.h"
#include "inet/mobility/contract/IMobility.h"
#include "inet/physicallayer/wireless/common/radio/packetlevel/Radio.h"
#include "inet/transportlayer/common/L4PortTag_m.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo.h"

namespace inet {
namespace aodv {

namespace {

std::string joinAddresses(const std::set<L3Address>& addresses)
{
    std::ostringstream os;
    bool first = true;
    for (const auto& address : addresses) {
        if (!first)
            os << "|";
        os << address;
        first = false;
    }
    return os.str();
}

std::string joinUnreachableNodes(const std::vector<UnreachableNode>& nodes)
{
    std::ostringstream os;
    bool first = true;
    for (const auto& node : nodes) {
        if (!first)
            os << "|";
        os << node.addr << ":" << node.seqNum;
        first = false;
    }
    return os.str();
}

double clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

double relu(double value)
{
    return value > 0 ? value : 0;
}

double sigmoid(double value)
{
    return 1.0 / (1.0 + std::exp(-value));
}

std::string trimCopy(const std::string& text)
{
    const char *whitespace = " \t\r\n";
    size_t begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos)
        return "";
    size_t end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

double softplus(double value)
{
    if (value > 20.0)
        return value;
    return std::log1p(std::exp(value));
}

double roundToDigits(double value, int digits)
{
    double scale = std::pow(10.0, std::max(0, digits));
    return std::round(value * scale) / scale;
}

} // namespace

Define_Module(Aodv);

const int KIND_DELAYEDSEND = 100;

void Aodv::initialize(int stage)
{
    if (stage == INITSTAGE_ROUTING_PROTOCOLS)
        addressType = getSelfIPAddress().getAddressType(); // needed for handleStartOperation()

    RoutingProtocolBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        lastBroadcastTime = SIMTIME_ZERO;
        nextRoutingTableSnapshotTime = SIMTIME_ZERO;
        rebootTime = SIMTIME_ZERO;
        rreqId = sequenceNum = 0;
        rreqCount = rerrCount = 0;
        host = getContainingNode(this);
        routingTable.reference(this, "routingTableModule", true);
        interfaceTable.reference(this, "interfaceTableModule", true);
        networkProtocol.reference(this, "networkProtocolModule", true);
        if (hasPar("pwd"))
            pwd = par("pwd").stdstringValue();
        enableRreqGraphLog = par("enableRreqGraphLog");
        enableRouteGraphLog = par("enableRouteGraphLog");
        enablePrecursorLog = par("enablePrecursorLog");
        enableRerrFanoutLog = par("enableRerrFanoutLog");
        enableRoutingTableSnapshotLog = par("enableRoutingTableSnapshotLog");
        enableSummary1sLog = par("enableSummary1sLog");
        cbrBasedRrepEnabled = par("cbrBasedRrepEnabled");
        cbrBasedRrepThreshold = par("cbrBasedRrepThreshold");
        cbrBasedRrepCompareMode = par("cbrBasedRrepCompareMode").stdstringValue();
        cbrBasedRrepRangeEnabled = par("cbrBasedRrepRangeEnabled");
        cbrBasedRrepLowThreshold = par("cbrBasedRrepLowThreshold");
        cbrBasedRrepHighThresholdForRange = par("cbrBasedRrepHighThresholdForRange");
        cbrBasedRandomThresholdEnabled = par("cbrBasedRandomThresholdEnabled");
        cbrBasedRandomThresholdUpdateInterval = par("cbrBasedRandomThresholdUpdateInterval");
        cbrBasedRandomLowMin = par("cbrBasedRandomLowMin");
        cbrBasedRandomLowMax = par("cbrBasedRandomLowMax");
        cbrBasedRandomHighMin = par("cbrBasedRandomHighMin");
        cbrBasedRandomHighMax = par("cbrBasedRandomHighMax");
        cbrBasedRandomMinGap = par("cbrBasedRandomMinGap");
        cbrBasedRrepDirectRouteBypassEnabled = par("cbrBasedRrepDirectRouteBypassEnabled");
        dlBasedRrepEnabled = par("dlBasedRrepEnabled");
        dlBasedRrepScoreThreshold = par("dlBasedRrepScoreThreshold");
        dlBasedRrepCompareMode = par("dlBasedRrepCompareMode").stdstringValue();
        dlBasedRrepFeatureSet = par("dlBasedRrepFeatureSet").stdstringValue();
        dlBasedRrepNeighborNorm = par("dlBasedRrepNeighborNorm");
        dlBasedRrepHopNorm = par("dlBasedRrepHopNorm");
        dlBasedRrepThresholdMin = par("dlBasedRrepThresholdMin");
        dlBasedRrepThresholdMax = par("dlBasedRrepThresholdMax");
        dlBasedRrepMinThresholdGap = par("dlBasedRrepMinThresholdGap");
        dlBasedRrepDirectThresholdOutputEnabled = par("dlBasedRrepDirectThresholdOutputEnabled");
        dlBasedRrepCustomArchitectureEnabled = par("dlBasedRrepCustomArchitectureEnabled");
        dlBasedRrepHidden1Size = par("dlBasedRrepHidden1Size");
        dlBasedRrepHidden2Size = par("dlBasedRrepHidden2Size");
        dlBasedRrepHidden3Size = par("dlBasedRrepHidden3Size");
        dlDirectThresholdRrepEnabled = par("dlDirectThresholdRrepEnabled");
        dlDirectThresholdRrepFeatureSet = par("dlDirectThresholdRrepFeatureSet").stdstringValue();
        dlDirectThresholdRrepNeighborNorm = par("dlDirectThresholdRrepNeighborNorm");
        dlDirectThresholdRrepHopNorm = par("dlDirectThresholdRrepHopNorm");
        dlDirectThresholdRrepThresholdMin = par("dlDirectThresholdRrepThresholdMin");
        dlDirectThresholdRrepThresholdMax = par("dlDirectThresholdRrepThresholdMax");
        dlDirectThresholdRrepMinThresholdGap = par("dlDirectThresholdRrepMinThresholdGap");
        dlDirectThresholdRrepInputStandardizationEnabled = par("dlDirectThresholdRrepInputStandardizationEnabled");
        dlDirectThresholdRrepOutputStandardizationEnabled = par("dlDirectThresholdRrepOutputStandardizationEnabled");
        dlDirectThresholdRrepHidden1Size = par("dlDirectThresholdRrepHidden1Size");
        dlDirectThresholdRrepHidden2Size = par("dlDirectThresholdRrepHidden2Size");
        dlDirectThresholdRrepHidden3Size = par("dlDirectThresholdRrepHidden3Size");
        dlBucketBasedRrepEnabled = par("dlBucketBasedRrepEnabled");
        dlBucketBasedRrepNeighborNorm = par("dlBucketBasedRrepNeighborNorm");
        dlBucketBasedRrepHopNorm = par("dlBucketBasedRrepHopNorm");
        dlBucketBasedRrepBucketRoundDigits = par("dlBucketBasedRrepBucketRoundDigits");
        dlBucketBasedRrepNearestFallbackEnabled = par("dlBucketBasedRrepNearestFallbackEnabled");
        dlBucketBasedRrepLookupTable = par("dlBucketBasedRrepLookupTable").stdstringValue();
        stateLookupBasedRrepEnabled = par("stateLookupBasedRrepEnabled");
        stateLookupBasedRrepNearestFallbackEnabled = par("stateLookupBasedRrepNearestFallbackEnabled");
        stateLookupBasedRrepCbrBinSize = par("stateLookupBasedRrepCbrBinSize");
        stateLookupBasedRrepNeighborBinSize = par("stateLookupBasedRrepNeighborBinSize");
        stateLookupBasedRrepHopMax = par("stateLookupBasedRrepHopMax");
        stateLookupBasedRrepPolicyCsvPath = par("stateLookupBasedRrepPolicyCsvPath").stdstringValue();
        treeBasedRrepEnabled = par("treeBasedRrepEnabled");
        treeBasedRrepFeatureSet = par("treeBasedRrepFeatureSet").stdstringValue();
        treeBasedRrepNeighborNorm = par("treeBasedRrepNeighborNorm");
        treeBasedRrepHopNorm = par("treeBasedRrepHopNorm");
        treeBasedRrepThresholdMin = par("treeBasedRrepThresholdMin");
        treeBasedRrepThresholdMax = par("treeBasedRrepThresholdMax");
        treeBasedRrepMinThresholdGap = par("treeBasedRrepMinThresholdGap");
        treeBasedRrepModel = par("treeBasedRrepModel").stdstringValue();
        treeBasedRrepLowModel = par("treeBasedRrepLowModel").stdstringValue();
        treeBasedRrepHighModel = par("treeBasedRrepHighModel").stdstringValue();
        cbrBasedRrepDelayEnabled = par("cbrBasedRrepDelayEnabled");
        cbrBasedRrepModerateThreshold = par("cbrBasedRrepModerateThreshold");
        cbrBasedRrepHighThreshold = par("cbrBasedRrepHighThreshold");
        cbrBasedRrepModerateDelay = par("cbrBasedRrepModerateDelay");
        cbrBasedRrepHighDelay = par("cbrBasedRrepHighDelay");
        cbrRrepMetricsEnabled = par("cbrRrepMetricsEnabled");
        cbrRrepDecisionLogEnabled = par("cbrRrepDecisionLogEnabled");
        dlDirectThresholdRrepDebugLogEnabled = par("dlDirectThresholdRrepDebugLogEnabled");
        cbrRouteCauseLogEnabled = par("cbrRouteCauseLogEnabled");
        transmissionFailureDiagnosisLogEnabled = par("transmissionFailureDiagnosisLogEnabled");
        useBdStationCount = par("useBdStationCount");
        if (dlBasedRrepEnabled)
            loadDlBasedRrepParameters();
        else {
            dlBasedRrepHiddenWeights.clear();
            dlBasedRrepHiddenBiases.clear();
            dlBasedRrepHidden2Weights.clear();
            dlBasedRrepHidden2Biases.clear();
            dlBasedRrepHidden3Weights.clear();
            dlBasedRrepHidden3Biases.clear();
            dlBasedRrepOutputWeights.clear();
            dlBasedRrepOutputBiases.clear();
        }
        if (dlDirectThresholdRrepEnabled)
            loadDlDirectThresholdRrepParameters();
        else {
            dlDirectThresholdRrepHiddenWeights.clear();
            dlDirectThresholdRrepHiddenBiases.clear();
            dlDirectThresholdRrepHidden2Weights.clear();
            dlDirectThresholdRrepHidden2Biases.clear();
            dlDirectThresholdRrepHidden3Weights.clear();
            dlDirectThresholdRrepHidden3Biases.clear();
            dlDirectThresholdRrepOutputWeights.clear();
            dlDirectThresholdRrepOutputBiases.clear();
        }

        if (dlBucketBasedRrepEnabled)
            loadDlBucketBasedRrepParameters();
        else
        {
            dlBucketBasedRrepEntriesByKey.clear();
            dlBucketBasedRrepEntries.clear();
        }

        if (stateLookupBasedRrepEnabled)
            loadStateLookupBasedRrepParameters();
        else {
            stateLookupBasedRrepEntriesByKey.clear();
            stateLookupBasedRrepEntries.clear();
        }

        if (treeBasedRrepEnabled)
            loadTreeBasedRrepParameters();
        else {
            treeBasedRrepLowEnsemble.clear();
            treeBasedRrepHighEnsemble.clear();
        }

        aodvUDPPort = par("udpPort");
        askGratuitousRREP = par("askGratuitousRREP");
        useHelloMessages = par("useHelloMessages");
        destinationOnlyFlag = par("destinationOnlyFlag");
        activeRouteTimeout = par("activeRouteTimeout");
        helloInterval = par("helloInterval");
        allowedHelloLoss = par("allowedHelloLoss");
        netDiameter = par("netDiameter");
        nodeTraversalTime = par("nodeTraversalTime");
        rerrRatelimit = par("rerrRatelimit");
        rreqRetries = par("rreqRetries");
        rreqRatelimit = par("rreqRatelimit");
        timeoutBuffer = par("timeoutBuffer");
        ttlStart = par("ttlStart");
        ttlIncrement = par("ttlIncrement");
        ttlThreshold = par("ttlThreshold");
        localAddTTL = par("localAddTTL");
        jitterPar = &par("jitter");
        periodicJitter = &par("periodicJitter");

        myRouteTimeout = par("myRouteTimeout");
        deletePeriod = par("deletePeriod");
        blacklistTimeout = par("blacklistTimeout");
        netTraversalTime = par("netTraversalTime");
        nextHopWait = par("nextHopWait");
        pathDiscoveryTime = par("pathDiscoveryTime");
        expungeTimer = new cMessage("ExpungeTimer");
        counterTimer = new cMessage("CounterTimer");
        rrepAckTimer = new cMessage("RrepAckTimer");
        blacklistTimer = new cMessage("BlackListTimer");
        if (useHelloMessages)
            helloMsgTimer = new cMessage("HelloMsgTimer");
    }
    else if (stage == INITSTAGE_ROUTING_PROTOCOLS) {
        networkProtocol->registerHook(0, this);
        host->subscribe(linkBrokenSignal, this);
        usingIpv6 = (routingTable->getRouterIdAsGeneric().getType() == L3Address::IPv6);
        if (cbrRrepDecisionLogEnabled)
            ensureCbrRrepDecisionLogFile();
    }
}

void Aodv::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (auto waitForRrep = dynamic_cast<WaitForRrep *>(msg))
            handleWaitForRREP(waitForRrep);
        else if (msg == helloMsgTimer)
            sendHelloMessagesIfNeeded();
        else if (msg == expungeTimer)
            expungeRoutes();
        else if (msg == counterTimer) {
            logSummary1s();
            logTransmissionFailureDiagnosis1s();
            logCbrRrepMetrics1s();
            rreqCount = rerrCount = 0;
            scheduleAfter(1, counterTimer);
        }
        else if (msg == rrepAckTimer)
            handleRREPACKTimer();
        else if (msg == blacklistTimer)
            handleBlackListTimer();
        else if (msg->getKind() == KIND_DELAYEDSEND) {
            auto timer = check_and_cast<PacketHolderMessage *>(msg);
            socket.send(timer->removeOwnedPacket());
            delete timer;
        }
        else
            throw cRuntimeError("Unknown self message");
    }
    else
        socket.processMessage(msg);
}

void Aodv::checkIpVersionAndPacketTypeCompatibility(AodvControlPacketType packetType)
{
    switch (packetType) {
    case RREQ:
    case RREP:
    case RERR:
    case RREPACK:
        if (usingIpv6)
            throw cRuntimeError("AODV Control Packet arrived with non-IPv6 packet type %d, but AODV configured for IPv6 routing", packetType);
        break;

    case RREQ_IPv6:
    case RREP_IPv6:
    case RERR_IPv6:
    case RREPACK_IPv6:
        if (!usingIpv6)
            throw cRuntimeError("AODV Control Packet arrived with IPv6 packet type %d, but AODV configured for non-IPv6 routing", packetType);
        break;

    default:
        throw cRuntimeError("AODV Control Packet arrived with undefined packet type: %d", packetType);
    }
}

void Aodv::processPacket(Packet *packet)
{
    L3Address sourceAddr = packet->getTag<L3AddressInd>()->getSrcAddress();
    // KLUDGE I added this -1 after TTL decrement has been moved in Ipv4
    unsigned int arrivalPacketTTL = packet->getTag<HopLimitInd>()->getHopLimit() - 1;
    const auto& aodvPacket = packet->popAtFront<AodvControlPacket>();
    // TODO aodvPacket->copyTags(*udpPacket);

    auto packetType = aodvPacket->getPacketType();
    switch (packetType) {
    case RREQ:
    case RREQ_IPv6:
        checkIpVersionAndPacketTypeCompatibility(packetType);
        handleRREQ(CHK(dynamicPtrCast<Rreq>(aodvPacket->dupShared())), sourceAddr, arrivalPacketTTL);
        delete packet;
        return;

    case RREP:
    case RREP_IPv6:
        checkIpVersionAndPacketTypeCompatibility(packetType);
        handleRREP(CHK(dynamicPtrCast<Rrep>(aodvPacket->dupShared())), sourceAddr);
        delete packet;
        return;

    case RERR:
    case RERR_IPv6:
        checkIpVersionAndPacketTypeCompatibility(packetType);
        handleRERR(CHK(dynamicPtrCast<const Rerr>(aodvPacket)), sourceAddr);
        delete packet;
        return;

    case RREPACK:
    case RREPACK_IPv6:
        checkIpVersionAndPacketTypeCompatibility(packetType);
        handleRREPACK(CHK(dynamicPtrCast<const RrepAck>(aodvPacket)), sourceAddr);
        delete packet;
        return;

    default:
        throw cRuntimeError("AODV Control Packet arrived with undefined packet type: %d", packetType);
    }
}

INetfilter::IHook::Result Aodv::ensureRouteForDatagram(Packet *datagram)
{
    const auto& networkHeader = getNetworkProtocolHeader(datagram);
    const L3Address& destAddr = networkHeader->getDestinationAddress();
    const L3Address& sourceAddr = networkHeader->getSourceAddress();

    if (destAddr.isBroadcast() || routingTable->isLocalAddress(destAddr) || destAddr.isMulticast())
        return ACCEPT;
    else {
        EV_INFO << "Finding route for source " << sourceAddr << " with destination " << destAddr << endl;
        IRoute *route = routingTable->findBestMatchingRoute(destAddr);
        AodvRouteData *routeData = route ? dynamic_cast<AodvRouteData *>(route->getProtocolData()) : nullptr;
        bool isActive = routeData && routeData->isActive();
        if (isActive && !route->getNextHopAsGeneric().isUnspecified()) {
            EV_INFO << "Active route found: " << route << endl;

            // Each time a route is used to forward a data packet, its Active Route
            // Lifetime field of the source, destination and the next hop on the
            // path to the destination is updated to be no less than the current
            // time plus ACTIVE_ROUTE_TIMEOUT.

            updateValidRouteLifeTime(destAddr, simTime() + activeRouteTimeout);
            updateValidRouteLifeTime(route->getNextHopAsGeneric(), simTime() + activeRouteTimeout);

            return ACCEPT;
        }
        else {
            bool isInactive = routeData && !routeData->isActive();
            // A node disseminates a RREQ when it determines that it needs a route
            // to a destination and does not have one available.  This can happen if
            // the destination is previously unknown to the node, or if a previously
            // valid route to the destination expires or is marked as invalid.

            EV_INFO << (isInactive ? "Inactive" : "Missing") << " route for destination " << destAddr << endl;

            delayDatagram(datagram);

            if (!hasOngoingRouteDiscovery(destAddr)) {
                // When a new route to the same destination is required at a later time
                // (e.g., upon route loss), the TTL in the RREQ IP header is initially
                // set to the Hop Count plus TTL_INCREMENT.
                if (isInactive)
                    startRouteDiscovery(destAddr, route->getMetric() + ttlIncrement);
                else
                    startRouteDiscovery(destAddr);
            }
            else
                EV_DETAIL << "Route discovery is in progress, originator " << getSelfIPAddress() << " target " << destAddr << endl;

            return QUEUE;
        }
    }
}

Aodv::Aodv()
{
}

bool Aodv::hasOngoingRouteDiscovery(const L3Address& target)
{
    return containsKey(waitForRREPTimers, target);
}

void Aodv::startRouteDiscovery(const L3Address& target, unsigned timeToLive)
{
    EV_INFO << "Starting route discovery with originator " << getSelfIPAddress() << " and destination " << target << endl;
    ASSERT(!hasOngoingRouteDiscovery(target));
    if (cbrRrepMetricsEnabled) {
        metricsRouteDiscoveryStartedCount++;
        metricsRouteDiscoveryStartTimes[target] = simTime();
        metricsRouteDiscoveryCandidateCounts[target] = 0;
    }
    auto rreq = createRREQ(target);
    addressToRreqRetries[target] = 0;
    sendRREQ(rreq, addressType->getBroadcastAddress(), timeToLive);
}

L3Address Aodv::getSelfIPAddress() const
{
    return routingTable->getRouterIdAsGeneric();
}

void Aodv::delayDatagram(Packet *datagram)
{
    const auto& networkHeader = getNetworkProtocolHeader(datagram);
    EV_DETAIL << "Queuing datagram, source " << networkHeader->getSourceAddress() << ", destination " << networkHeader->getDestinationAddress() << endl;
    const L3Address& target = networkHeader->getDestinationAddress();
    targetAddressToDelayedPackets.insert(std::pair<L3Address, Packet *>(target, datagram));
}

void Aodv::sendRREQ(const Ptr<Rreq>& rreq, const L3Address& destAddr, unsigned int timeToLive)
{
    // In an expanding ring search, the originating node initially uses a TTL =
    // TTL_START in the RREQ packet IP header and sets the timeout for
    // receiving a RREP to RING_TRAVERSAL_TIME milliseconds.
    // RING_TRAVERSAL_TIME is calculated as described in section 10.  The
    // TTL_VALUE used in calculating RING_TRAVERSAL_TIME is set equal to the
    // value of the TTL field in the IP header.  If the RREQ times out
    // without a corresponding RREP, the originator broadcasts the RREQ
    // again with the TTL incremented by TTL_INCREMENT.  This continues
    // until the TTL set in the RREQ reaches TTL_THRESHOLD, beyond which a
    // TTL = NET_DIAMETER is used for each attempt.

    if (rreqCount >= rreqRatelimit) {
        EV_WARN << "A node should not originate more than RREQ_RATELIMIT RREQ messages per second. Canceling sending RREQ" << endl;
        return;
    }

    auto rrepTimer = waitForRREPTimers.find(rreq->getDestAddr());
    WaitForRrep *rrepTimerMsg = nullptr;
    if (rrepTimer != waitForRREPTimers.end()) {
        rrepTimerMsg = rrepTimer->second;
        unsigned int lastTTL = rrepTimerMsg->getLastTTL();
        rrepTimerMsg->setDestAddr(rreq->getDestAddr());

        // The Hop Count stored in an invalid routing table entry indicates the
        // last known hop count to that destination in the routing table.  When
        // a new route to the same destination is required at a later time
        // (e.g., upon route loss), the TTL in the RREQ IP header is initially
        // set to the Hop Count plus TTL_INCREMENT.  Thereafter, following each
        // timeout the TTL is incremented by TTL_INCREMENT until TTL =
        // TTL_THRESHOLD is reached.  Beyond this TTL = NET_DIAMETER is used.
        // Once TTL = NET_DIAMETER, the timeout for waiting for the RREP is set
        // to NET_TRAVERSAL_TIME, as specified in section 6.3.

        if (timeToLive != 0) {
            rrepTimerMsg->setLastTTL(timeToLive);
            rrepTimerMsg->setFromInvalidEntry(true);
            cancelEvent(rrepTimerMsg);
        }
        else if (lastTTL + ttlIncrement < ttlThreshold) {
            ASSERT(!rrepTimerMsg->isScheduled());
            timeToLive = lastTTL + ttlIncrement;
            rrepTimerMsg->setLastTTL(lastTTL + ttlIncrement);
        }
        else {
            ASSERT(!rrepTimerMsg->isScheduled());
            timeToLive = netDiameter;
            rrepTimerMsg->setLastTTL(netDiameter);
        }
    }
    else {
        rrepTimerMsg = new WaitForRrep();
        waitForRREPTimers[rreq->getDestAddr()] = rrepTimerMsg;
        ASSERT(hasOngoingRouteDiscovery(rreq->getDestAddr()));

        timeToLive = ttlStart;
        rrepTimerMsg->setLastTTL(ttlStart);
        rrepTimerMsg->setFromInvalidEntry(false);
        rrepTimerMsg->setDestAddr(rreq->getDestAddr());
    }

    // Each time, the timeout for receiving a RREP is RING_TRAVERSAL_TIME.
    simtime_t ringTraversalTime = 2.0 * nodeTraversalTime * (timeToLive + timeoutBuffer);
    scheduleAfter(ringTraversalTime, rrepTimerMsg);

    //std::cout << "Sending a Route Request with target " << rreq->getDestAddr() << " and TTL= " << timeToLive << endl;


    // Keep route discovery unchanged; MAC-level repetition is applied later for broadcast RREQs.
    simtime_t baseDelay = SimTime((double)*jitterPar, SIMTIME_S);
    /*
    appendAodvMetric("aodv_control_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RREQ_SEND, target=" + rreq->getDestAddr().str() +
            ", originator=" + rreq->getOriginatorAddr().str() +
            ", ttl=" + std::to_string(timeToLive) +
            ", rreqId=" + std::to_string(rreq->getRreqId()) +
            ", retryCount=" + std::to_string(addressToRreqRetries[rreq->getDestAddr()]) +
            ", jitter=" + baseDelay.str());*/
    sendAODVPacket(rreq, destAddr, timeToLive, baseDelay.dbl());


    rreqCount++;
}

void Aodv::sendRREP(const Ptr<Rrep>& rrep, const L3Address& destAddr, unsigned int timeToLive, simtime_t delay)
{
    EV_INFO << "Sending Route Reply to " << destAddr << endl;
    /*
    appendAodvMetric("aodv_control_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RREP_SEND, target=" + destAddr.str() +
            ", originator=" + rrep->getOriginatorAddr().str() +
            ", destination=" + rrep->getDestAddr().str() +
            ", hopCount=" + std::to_string(rrep->getHopCount()) +
            ", ttl=" + std::to_string(timeToLive));*/

    // When any node transmits a RREP, the precursor list for the
    // corresponding destination node is updated by adding to it
    // the next hop node to which the RREP is forwarded.

    IRoute *destRoute = routingTable->findBestMatchingRoute(destAddr);
    if (destRoute == nullptr) {
        EV_WARN << "Cannot send RREP to " << destAddr << ": no matching route exists anymore" << endl;
        return;
    }
    const L3Address& nextHop = destRoute->getNextHopAsGeneric();
    AodvRouteData *destRouteData = check_and_cast<AodvRouteData *>(destRoute->getProtocolData());
    destRouteData->addPrecursor(nextHop);
    logPrecursorAddition("SEND_RREP_DEST", destRoute->getDestinationAsGeneric(), nextHop, destRouteData->getPrecursorList());

    // The node we received the Route Request for is our neighbor,
    // it is probably an unidirectional link
    if (destRoute->getMetric() == 1) {
        // It is possible that a RREP transmission may fail, especially if the
        // RREQ transmission triggering the RREP occurs over a unidirectional
        // link.

        rrep->setAckRequiredFlag(true);

        // when a node detects that its transmission of a RREP message has failed,
        // it remembers the next-hop of the failed RREP in a "blacklist" set.

        failedNextHop = nextHop;

        rescheduleAfter(nextHopWait, rrepAckTimer);
    }
    sendAODVPacket(rrep, nextHop, timeToLive, delay.dbl());
}

simtime_t Aodv::computeIntermediateRrepDelay(double localCbr, bool isDirectRouteToDestination) const
{
    if (!cbrBasedRrepDelayEnabled)
        return SIMTIME_ZERO;
    if (cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)
        return SIMTIME_ZERO;
    if (localCbr >= cbrBasedRrepHighThreshold)
        return cbrBasedRrepHighDelay;
    if (localCbr >= cbrBasedRrepModerateThreshold)
        return cbrBasedRrepModerateDelay;
    return SIMTIME_ZERO;
}

bool Aodv::shouldBlockByMode(double value, double threshold, const std::string& mode) const
{
    if (mode == "low")
        return value < threshold;
    if (mode == "high")
        return value > threshold;
    throw cRuntimeError("Unsupported compare mode '%s'. Use 'low' or 'high'.", mode.c_str());
}

const char *Aodv::describeModeRelation(const std::string& mode) const
{
    if (mode == "low")
        return "below";
    if (mode == "high")
        return "above";
    return "outside";
}

std::pair<double, double> Aodv::getActiveCbrThresholdRange()
{
    if (!cbrBasedRandomThresholdEnabled)
        return {cbrBasedRrepLowThreshold, cbrBasedRrepHighThresholdForRange};

    int currentEpoch = static_cast<int>(std::floor(simTime().dbl() / cbrBasedRandomThresholdUpdateInterval.dbl()));
    if (currentEpoch != cbrBasedRandomThresholdEpoch) {
        double low = 0;
        double high = 0;
        do {
            low = uniform(cbrBasedRandomLowMin, cbrBasedRandomLowMax);
            high = uniform(cbrBasedRandomHighMin, cbrBasedRandomHighMax);
        } while (high - low < cbrBasedRandomMinGap);
        cbrBasedRandomActiveLowThreshold = low;
        cbrBasedRandomActiveHighThreshold = high;
        cbrBasedRandomThresholdEpoch = currentEpoch;
    }

    return {cbrBasedRandomActiveLowThreshold, cbrBasedRandomActiveHighThreshold};
}

bool Aodv::isOutsideConfiguredCbrRange(double localCbr, double& activeLowThreshold, double& activeHighThreshold)
{
    if (!cbrBasedRrepRangeEnabled && !cbrBasedRandomThresholdEnabled)
        return false;

    auto activeRange = getActiveCbrThresholdRange();
    activeLowThreshold = activeRange.first;
    activeHighThreshold = activeRange.second;
    return !(activeLowThreshold < localCbr && localCbr < activeHighThreshold);
}

std::vector<double> Aodv::parseDoubleList(const char *text) const
{
    std::vector<double> values;
    std::istringstream input(text ? text : "");
    double value = 0;
    while (input >> value)
        values.push_back(value);
    return values;
}

size_t Aodv::getRrepFeatureInputSize(const std::string& featureSet) const
{
    if (featureSet == "basic4")
        return 4;
    if (featureSet == "raw4")
        return 4;
    if (featureSet == "local6")
        return 6;
    if (featureSet == "local10")
        return 10;
    throw cRuntimeError("featureSet must be one of 'basic4', 'raw4', 'local6', or 'local10'");
}

std::vector<double> Aodv::buildRrepFeatureInputs(const std::string& featureSet, int neighborNormDiv, int hopNormDiv, double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    double localCbrNorm = clamp01(localCbr / 100.0);
    double neighborNorm = clamp01(static_cast<double>(neighborCount) / neighborNormDiv);
    double hopNorm = clamp01(static_cast<double>(hopCount) / hopNormDiv);
    double diagLocalCbrNorm = localCbrNorm;
    double rreqReceived = static_cast<double>(metricsRreqReceivedCount);
    double rrepCandidates = static_cast<double>(metricsRrepCandidateCount);
    double rrepBlockRate = metricsRrepCandidateCount > 0 ? static_cast<double>(metricsRrepBlockedCount) / metricsRrepCandidateCount : 0.0;
    double routeInvalidate = static_cast<double>(diagnosisRouteInvalidateCount);
    double routeExpireInactive = static_cast<double>(diagnosisRouteExpireInactiveCount);
    double routeDelete = static_cast<double>(diagnosisRouteDeleteCount);
    double rerrOriginated = static_cast<double>(diagnosisRerrOriginatedCount);

    if (featureSet == "basic4") {
        return {
            localCbrNorm,
            neighborNorm,
            hopNorm,
            isDirectRouteToDestination ? 1.0 : 0.0
        };
    }

    if (featureSet == "raw4") {
        return {
            localCbr,
            static_cast<double>(neighborCount),
            static_cast<double>(hopCount),
            isDirectRouteToDestination ? 1.0 : 0.0
        };
    }

    if (featureSet == "local6") {
        return {
            diagLocalCbrNorm,
            rreqReceived,
            routeExpireInactive,
            localCbrNorm,
            routeInvalidate,
            rerrOriginated
        };
    }

    if (featureSet == "local10") {
        return {
            diagLocalCbrNorm,
            rreqReceived,
            routeExpireInactive,
            localCbrNorm,
            routeInvalidate,
            routeDelete,
            rerrOriginated,
            rrepCandidates,
            rrepBlockRate,
            neighborNorm
        };
    }

    throw cRuntimeError("Unsupported featureSet '%s'", featureSet.c_str());
}

size_t Aodv::getDlBasedRrepInputSize() const
{
    return getRrepFeatureInputSize(dlBasedRrepFeatureSet);
}

std::vector<double> Aodv::buildDlBasedRrepInputs(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    return buildRrepFeatureInputs(dlBasedRrepFeatureSet, dlBasedRrepNeighborNorm, dlBasedRrepHopNorm, localCbr, neighborCount, hopCount, isDirectRouteToDestination);
}

Aodv::TreeBasedRrepEnsemble Aodv::parseTreeBasedRrepEnsemble(const std::string& text, size_t inputSize, const char *fieldName) const
{
    TreeBasedRrepEnsemble ensemble;
    std::stringstream treeStream(text);
    std::string treeText;
    while (std::getline(treeStream, treeText, '#')) {
        if (treeText.empty())
            continue;

        TreeBasedRrepTree tree;
        std::stringstream nodeStream(treeText);
        std::string nodeText;
        while (std::getline(nodeStream, nodeText, ';')) {
            if (nodeText.empty())
                continue;

            std::stringstream tokenStream(nodeText);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(tokenStream, token, ','))
                tokens.push_back(token);

            if (tokens.empty())
                continue;

            TreeBasedRrepNode node;
            if (tokens[0] == "L") {
                if (tokens.size() != 2)
                    throw cRuntimeError("%s leaf node must be formatted as L,value", fieldName);
                node.isLeaf = true;
                node.value = std::stod(tokens[1]);
            }
            else if (tokens[0] == "I") {
                if (tokens.size() != 5)
                    throw cRuntimeError("%s internal node must be formatted as I,feature,threshold,left,right", fieldName);
                node.isLeaf = false;
                node.featureIndex = std::stoi(tokens[1]);
                node.threshold = std::stod(tokens[2]);
                node.leftIndex = std::stoi(tokens[3]);
                node.rightIndex = std::stoi(tokens[4]);
                if (node.featureIndex < 0 || static_cast<size_t>(node.featureIndex) >= inputSize)
                    throw cRuntimeError("%s feature index %d is out of range for input size %zu", fieldName, node.featureIndex, inputSize);
            }
            else
                throw cRuntimeError("%s node must start with L or I", fieldName);

            tree.push_back(node);
        }

        if (tree.empty())
            continue;

        for (size_t index = 0; index < tree.size(); ++index) {
            const auto& node = tree[index];
            if (!node.isLeaf) {
                if (node.leftIndex < 0 || static_cast<size_t>(node.leftIndex) >= tree.size() ||
                    node.rightIndex < 0 || static_cast<size_t>(node.rightIndex) >= tree.size())
                {
                    throw cRuntimeError("%s node %zu references an invalid child index", fieldName, index);
                }
            }
        }
        ensemble.push_back(tree);
    }

    if (ensemble.empty())
        throw cRuntimeError("%s did not contain any valid trees", fieldName);

    return ensemble;
}

double Aodv::evaluateTreeBasedRrepEnsemble(const TreeBasedRrepEnsemble& ensemble, const std::vector<double>& inputs, const char *fieldName) const
{
    if (ensemble.empty())
        throw cRuntimeError("%s ensemble is empty", fieldName);

    double sum = 0.0;
    for (const auto& tree : ensemble) {
        int nodeIndex = 0;
        int guard = 0;
        while (true) {
            if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= tree.size())
                throw cRuntimeError("%s traversal reached invalid node index %d", fieldName, nodeIndex);
            const auto& node = tree[static_cast<size_t>(nodeIndex)];
            if (node.isLeaf) {
                sum += node.value;
                break;
            }
            double featureValue = inputs[static_cast<size_t>(node.featureIndex)];
            nodeIndex = featureValue <= node.threshold ? node.leftIndex : node.rightIndex;
            guard++;
            if (guard > 100000)
                throw cRuntimeError("%s traversal exceeded safety guard", fieldName);
        }
    }

    return sum / static_cast<double>(ensemble.size());
}

void Aodv::loadDlBasedRrepParameters()
{
    constexpr size_t outputSize = 2;
    size_t inputSize = getDlBasedRrepInputSize();
    int enabledPredictors = (dlBasedRrepEnabled ? 1 : 0) + (dlDirectThresholdRrepEnabled ? 1 : 0) + (dlBucketBasedRrepEnabled ? 1 : 0) + (stateLookupBasedRrepEnabled ? 1 : 0) + (treeBasedRrepEnabled ? 1 : 0);
    if (enabledPredictors > 1)
        throw cRuntimeError("Only one of dlBasedRrepEnabled, dlDirectThresholdRrepEnabled, dlBucketBasedRrepEnabled, stateLookupBasedRrepEnabled, or treeBasedRrepEnabled may be enabled at a time");

    dlBasedRrepHiddenWeights = parseDoubleList(par("dlBasedRrepHiddenWeights").stringValue());
    dlBasedRrepHiddenBiases = parseDoubleList(par("dlBasedRrepHiddenBiases").stringValue());
    dlBasedRrepHidden2Weights = parseDoubleList(par("dlBasedRrepHidden2Weights").stringValue());
    dlBasedRrepHidden2Biases = parseDoubleList(par("dlBasedRrepHidden2Biases").stringValue());
    dlBasedRrepHidden3Weights = parseDoubleList(par("dlBasedRrepHidden3Weights").stringValue());
    dlBasedRrepHidden3Biases = parseDoubleList(par("dlBasedRrepHidden3Biases").stringValue());
    dlBasedRrepOutputWeights = parseDoubleList(par("dlBasedRrepOutputWeights").stringValue());
    dlBasedRrepOutputBiases = parseDoubleList(par("dlBasedRrepOutputBias").stringValue());

    if (cbrBasedRrepCompareMode != "low" && cbrBasedRrepCompareMode != "high")
        throw cRuntimeError("cbrBasedRrepCompareMode must be 'low' or 'high'");
    if (cbrBasedRrepRangeEnabled && cbrBasedRrepLowThreshold >= cbrBasedRrepHighThresholdForRange)
        throw cRuntimeError("cbrBasedRrepLowThreshold must be smaller than cbrBasedRrepHighThresholdForRange");
    if (cbrBasedRandomThresholdEnabled) {
        if (cbrBasedRandomThresholdUpdateInterval <= SIMTIME_ZERO)
            throw cRuntimeError("cbrBasedRandomThresholdUpdateInterval must be positive");
        if (cbrBasedRandomLowMin >= cbrBasedRandomLowMax)
            throw cRuntimeError("cbrBasedRandomLowMin must be smaller than cbrBasedRandomLowMax");
        if (cbrBasedRandomHighMin >= cbrBasedRandomHighMax)
            throw cRuntimeError("cbrBasedRandomHighMin must be smaller than cbrBasedRandomHighMax");
        if (cbrBasedRandomLowMax >= cbrBasedRandomHighMax)
            throw cRuntimeError("cbrBasedRandomLowMax must be smaller than cbrBasedRandomHighMax");
        if (cbrBasedRandomMinGap <= 0)
            throw cRuntimeError("cbrBasedRandomMinGap must be positive");
        if (cbrBasedRandomHighMax - cbrBasedRandomLowMin < cbrBasedRandomMinGap)
            throw cRuntimeError("Random threshold range cannot satisfy cbrBasedRandomMinGap");
    }

    if (!dlBasedRrepEnabled)
        return;

    if (dlBasedRrepCompareMode != "low" && dlBasedRrepCompareMode != "high")
        throw cRuntimeError("dlBasedRrepCompareMode must be 'low' or 'high'");
    if (dlBasedRrepFeatureSet != "basic4" && dlBasedRrepFeatureSet != "local6" && dlBasedRrepFeatureSet != "local10")
        throw cRuntimeError("dlBasedRrepFeatureSet must be one of 'basic4', 'local6', or 'local10'");

    if (dlBasedRrepNeighborNorm <= 0)
        throw cRuntimeError("dlBasedRrepNeighborNorm must be positive");
    if (dlBasedRrepHopNorm <= 0)
        throw cRuntimeError("dlBasedRrepHopNorm must be positive");
    if (dlBasedRrepThresholdMin >= dlBasedRrepThresholdMax)
        throw cRuntimeError("dlBasedRrepThresholdMin must be smaller than dlBasedRrepThresholdMax");
    if (dlBasedRrepMinThresholdGap < 0)
        throw cRuntimeError("dlBasedRrepMinThresholdGap must be non-negative");
    if (dlBasedRrepMinThresholdGap > dlBasedRrepThresholdMax - dlBasedRrepThresholdMin)
        throw cRuntimeError("dlBasedRrepMinThresholdGap must not exceed the threshold range");
    if (dlBasedRrepHidden1Size <= 0)
        throw cRuntimeError("dlBasedRrepHidden1Size must be positive");
    if (dlBasedRrepHidden2Size <= 0)
        throw cRuntimeError("dlBasedRrepHidden2Size must be positive");
    if (dlBasedRrepDirectThresholdOutputEnabled && dlBasedRrepHidden3Size <= 0)
        throw cRuntimeError("dlBasedRrepHidden3Size must be positive when dlBasedRrepDirectThresholdOutputEnabled is true");
    size_t hidden1Size = static_cast<size_t>(dlBasedRrepHidden1Size);
    size_t hidden2Size = static_cast<size_t>(dlBasedRrepHidden2Size);
    size_t hidden3Size = static_cast<size_t>(dlBasedRrepHidden3Size);
    if (!dlBasedRrepCustomArchitectureEnabled) {
        hidden1Size = 32;
        hidden2Size = 16;
        if (!dlBasedRrepDirectThresholdOutputEnabled)
            hidden3Size = 0;
    }
    if (dlBasedRrepHiddenWeights.size() != inputSize * hidden1Size)
        throw cRuntimeError("dlBasedRrepHiddenWeights must contain exactly %zu values (%zux%zu)", inputSize * hidden1Size, inputSize, hidden1Size);
    if (dlBasedRrepHiddenBiases.size() != hidden1Size)
        throw cRuntimeError("dlBasedRrepHiddenBiases must contain exactly %zu values", hidden1Size);
    if (dlBasedRrepHidden2Weights.size() != hidden1Size * hidden2Size)
        throw cRuntimeError("dlBasedRrepHidden2Weights must contain exactly %zu values (%zux%zu)", hidden1Size * hidden2Size, hidden1Size, hidden2Size);
    if (dlBasedRrepHidden2Biases.size() != hidden2Size)
        throw cRuntimeError("dlBasedRrepHidden2Biases must contain exactly %zu values", hidden2Size);
    if (dlBasedRrepDirectThresholdOutputEnabled) {
        if (dlBasedRrepHidden3Weights.size() != hidden2Size * hidden3Size)
            throw cRuntimeError("dlBasedRrepHidden3Weights must contain exactly %zu values (%zux%zu)", hidden2Size * hidden3Size, hidden2Size, hidden3Size);
        if (dlBasedRrepHidden3Biases.size() != hidden3Size)
            throw cRuntimeError("dlBasedRrepHidden3Biases must contain exactly %zu values", hidden3Size);
        if (dlBasedRrepOutputWeights.size() != hidden3Size * outputSize)
            throw cRuntimeError("dlBasedRrepOutputWeights must contain exactly %zu values (%zux%zu)", hidden3Size * outputSize, hidden3Size, outputSize);
    }
    else {
        if (!dlBasedRrepHidden3Weights.empty() && !(dlBasedRrepHidden3Weights.size() == 1 && dlBasedRrepHidden3Weights[0] == 0.0))
            throw cRuntimeError("dlBasedRrepHidden3Weights must be empty when dlBasedRrepDirectThresholdOutputEnabled is false");
        if (!dlBasedRrepHidden3Biases.empty() && !(dlBasedRrepHidden3Biases.size() == 1 && dlBasedRrepHidden3Biases[0] == 0.0))
            throw cRuntimeError("dlBasedRrepHidden3Biases must be empty when dlBasedRrepDirectThresholdOutputEnabled is false");
        if (dlBasedRrepOutputWeights.size() != hidden2Size * outputSize)
            throw cRuntimeError("dlBasedRrepOutputWeights must contain exactly %zu values (%zux%zu)", hidden2Size * outputSize, hidden2Size, outputSize);
    }
    if (dlBasedRrepOutputBiases.size() != outputSize)
        throw cRuntimeError("dlBasedRrepOutputBias must contain exactly %zu values", outputSize);
}

void Aodv::loadDlDirectThresholdRrepParameters()
{
    constexpr size_t outputSize = 2;
    size_t inputSize = getRrepFeatureInputSize(dlDirectThresholdRrepFeatureSet);
    int enabledPredictors = (dlBasedRrepEnabled ? 1 : 0) + (dlDirectThresholdRrepEnabled ? 1 : 0) + (dlBucketBasedRrepEnabled ? 1 : 0) + (stateLookupBasedRrepEnabled ? 1 : 0) + (treeBasedRrepEnabled ? 1 : 0);
    if (enabledPredictors > 1)
        throw cRuntimeError("Only one of dlBasedRrepEnabled, dlDirectThresholdRrepEnabled, dlBucketBasedRrepEnabled, stateLookupBasedRrepEnabled, or treeBasedRrepEnabled may be enabled at a time");

    dlDirectThresholdRrepHiddenWeights = parseDoubleList(par("dlDirectThresholdRrepHiddenWeights").stringValue());
    dlDirectThresholdRrepHiddenBiases = parseDoubleList(par("dlDirectThresholdRrepHiddenBiases").stringValue());
    dlDirectThresholdRrepHidden2Weights = parseDoubleList(par("dlDirectThresholdRrepHidden2Weights").stringValue());
    dlDirectThresholdRrepHidden2Biases = parseDoubleList(par("dlDirectThresholdRrepHidden2Biases").stringValue());
    dlDirectThresholdRrepHidden3Weights = parseDoubleList(par("dlDirectThresholdRrepHidden3Weights").stringValue());
    dlDirectThresholdRrepHidden3Biases = parseDoubleList(par("dlDirectThresholdRrepHidden3Biases").stringValue());
    dlDirectThresholdRrepOutputWeights = parseDoubleList(par("dlDirectThresholdRrepOutputWeights").stringValue());
    dlDirectThresholdRrepOutputBiases = parseDoubleList(par("dlDirectThresholdRrepOutputBias").stringValue());
    dlDirectThresholdRrepInputMeans = parseDoubleList(par("dlDirectThresholdRrepInputMean").stringValue());
    dlDirectThresholdRrepInputScales = parseDoubleList(par("dlDirectThresholdRrepInputScale").stringValue());
    dlDirectThresholdRrepOutputMeans = parseDoubleList(par("dlDirectThresholdRrepOutputMean").stringValue());
    dlDirectThresholdRrepOutputScales = parseDoubleList(par("dlDirectThresholdRrepOutputScale").stringValue());

    if (!dlDirectThresholdRrepEnabled)
        return;

    if (dlDirectThresholdRrepFeatureSet != "basic4" && dlDirectThresholdRrepFeatureSet != "raw4" && dlDirectThresholdRrepFeatureSet != "local6" && dlDirectThresholdRrepFeatureSet != "local10")
        throw cRuntimeError("dlDirectThresholdRrepFeatureSet must be one of 'basic4', 'raw4', 'local6', or 'local10'");
    if (dlDirectThresholdRrepNeighborNorm <= 0)
        throw cRuntimeError("dlDirectThresholdRrepNeighborNorm must be positive");
    if (dlDirectThresholdRrepHopNorm <= 0)
        throw cRuntimeError("dlDirectThresholdRrepHopNorm must be positive");
    if (dlDirectThresholdRrepThresholdMin >= dlDirectThresholdRrepThresholdMax)
        throw cRuntimeError("dlDirectThresholdRrepThresholdMin must be smaller than dlDirectThresholdRrepThresholdMax");
    if (dlDirectThresholdRrepMinThresholdGap < 0)
        throw cRuntimeError("dlDirectThresholdRrepMinThresholdGap must be non-negative");
    if (dlDirectThresholdRrepMinThresholdGap > dlDirectThresholdRrepThresholdMax - dlDirectThresholdRrepThresholdMin)
        throw cRuntimeError("dlDirectThresholdRrepMinThresholdGap must not exceed the threshold range");
    if (dlDirectThresholdRrepHidden1Size <= 0)
        throw cRuntimeError("dlDirectThresholdRrepHidden1Size must be positive");
    if (dlDirectThresholdRrepHidden2Size <= 0)
        throw cRuntimeError("dlDirectThresholdRrepHidden2Size must be positive");
    if (dlDirectThresholdRrepHidden3Size <= 0)
        throw cRuntimeError("dlDirectThresholdRrepHidden3Size must be positive");

    size_t hidden1Size = static_cast<size_t>(dlDirectThresholdRrepHidden1Size);
    size_t hidden2Size = static_cast<size_t>(dlDirectThresholdRrepHidden2Size);
    size_t hidden3Size = static_cast<size_t>(dlDirectThresholdRrepHidden3Size);
    if (dlDirectThresholdRrepInputStandardizationEnabled) {
        if (dlDirectThresholdRrepInputMeans.size() != inputSize)
            throw cRuntimeError("dlDirectThresholdRrepInputMean must contain exactly %zu values", inputSize);
        if (dlDirectThresholdRrepInputScales.size() != inputSize)
            throw cRuntimeError("dlDirectThresholdRrepInputScale must contain exactly %zu values", inputSize);
        for (size_t i = 0; i < inputSize; ++i) {
            if (dlDirectThresholdRrepInputScales[i] == 0.0)
                throw cRuntimeError("dlDirectThresholdRrepInputScale contains zero at index %zu", i);
        }
    }
    if (dlDirectThresholdRrepOutputStandardizationEnabled) {
        if (dlDirectThresholdRrepOutputMeans.size() != outputSize)
            throw cRuntimeError("dlDirectThresholdRrepOutputMean must contain exactly %zu values", outputSize);
        if (dlDirectThresholdRrepOutputScales.size() != outputSize)
            throw cRuntimeError("dlDirectThresholdRrepOutputScale must contain exactly %zu values", outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            if (dlDirectThresholdRrepOutputScales[i] == 0.0)
                throw cRuntimeError("dlDirectThresholdRrepOutputScale contains zero at index %zu", i);
        }
    }
    if (dlDirectThresholdRrepHiddenWeights.size() != inputSize * hidden1Size)
        throw cRuntimeError("dlDirectThresholdRrepHiddenWeights must contain exactly %zu values (%zux%zu)", inputSize * hidden1Size, inputSize, hidden1Size);
    if (dlDirectThresholdRrepHiddenBiases.size() != hidden1Size)
        throw cRuntimeError("dlDirectThresholdRrepHiddenBiases must contain exactly %zu values", hidden1Size);
    if (dlDirectThresholdRrepHidden2Weights.size() != hidden1Size * hidden2Size)
        throw cRuntimeError("dlDirectThresholdRrepHidden2Weights must contain exactly %zu values (%zux%zu)", hidden1Size * hidden2Size, hidden1Size, hidden2Size);
    if (dlDirectThresholdRrepHidden2Biases.size() != hidden2Size)
        throw cRuntimeError("dlDirectThresholdRrepHidden2Biases must contain exactly %zu values", hidden2Size);
    if (dlDirectThresholdRrepHidden3Weights.size() != hidden2Size * hidden3Size)
        throw cRuntimeError("dlDirectThresholdRrepHidden3Weights must contain exactly %zu values (%zux%zu)", hidden2Size * hidden3Size, hidden2Size, hidden3Size);
    if (dlDirectThresholdRrepHidden3Biases.size() != hidden3Size)
        throw cRuntimeError("dlDirectThresholdRrepHidden3Biases must contain exactly %zu values", hidden3Size);
    if (dlDirectThresholdRrepOutputWeights.size() != hidden3Size * outputSize)
        throw cRuntimeError("dlDirectThresholdRrepOutputWeights must contain exactly %zu values (%zux%zu)", hidden3Size * outputSize, hidden3Size, outputSize);
    if (dlDirectThresholdRrepOutputBiases.size() != outputSize)
        throw cRuntimeError("dlDirectThresholdRrepOutputBias must contain exactly %zu values", outputSize);
}

void Aodv::loadDlBucketBasedRrepParameters()
{
    dlBucketBasedRrepEntriesByKey.clear();
    dlBucketBasedRrepEntries.clear();

    int enabledPredictors = (dlBasedRrepEnabled ? 1 : 0) + (dlDirectThresholdRrepEnabled ? 1 : 0) + (dlBucketBasedRrepEnabled ? 1 : 0) + (treeBasedRrepEnabled ? 1 : 0);
    if (enabledPredictors > 1)
        throw cRuntimeError("Only one of dlBasedRrepEnabled, dlDirectThresholdRrepEnabled, dlBucketBasedRrepEnabled, or treeBasedRrepEnabled may be enabled at a time");

    if (!dlBucketBasedRrepEnabled)
        return;

    if (dlBucketBasedRrepNeighborNorm <= 0)
        throw cRuntimeError("dlBucketBasedRrepNeighborNorm must be positive");
    if (dlBucketBasedRrepHopNorm <= 0)
        throw cRuntimeError("dlBucketBasedRrepHopNorm must be positive");
    if (dlBucketBasedRrepBucketRoundDigits < 0 || dlBucketBasedRrepBucketRoundDigits > 6)
        throw cRuntimeError("dlBucketBasedRrepBucketRoundDigits must be between 0 and 6");
    if (dlBucketBasedRrepLookupTable.empty())
        throw cRuntimeError("dlBucketBasedRrepLookupTable must not be empty when dlBucketBasedRrepEnabled is true");

    std::stringstream entries(dlBucketBasedRrepLookupTable);
    std::string entryText;
    while (std::getline(entries, entryText, ';')) {
        if (entryText.empty())
            continue;

        auto eqPos = entryText.find('=');
        if (eqPos == std::string::npos)
            throw cRuntimeError("Invalid dlBucketBasedRrepLookupTable entry '%s': missing '='", entryText.c_str());

        std::string key = entryText.substr(0, eqPos);
        std::string pairText = entryText.substr(eqPos + 1);
        auto commaPos = pairText.find(',');
        if (commaPos == std::string::npos)
            throw cRuntimeError("Invalid dlBucketBasedRrepLookupTable entry '%s': missing ',' in threshold pair", entryText.c_str());

        std::string lowText = pairText.substr(0, commaPos);
        std::string highText = pairText.substr(commaPos + 1);
        double lowThreshold = std::stod(lowText);
        double highThreshold = std::stod(highText);
        if (!(lowThreshold < highThreshold))
            throw cRuntimeError("Invalid dlBucketBasedRrepLookupTable entry '%s': low threshold must be smaller than high threshold", entryText.c_str());

        std::stringstream keyStream(key);
        std::string part;
        std::vector<double> stateValues;
        while (std::getline(keyStream, part, '|')) {
            if (part.empty())
                throw cRuntimeError("Invalid dlBucketBasedRrepLookupTable key '%s'", key.c_str());
            stateValues.push_back(std::stod(part));
        }
        if (stateValues.size() != 4)
            throw cRuntimeError("Invalid dlBucketBasedRrepLookupTable key '%s': expected four state values", key.c_str());

        DlBucketBasedRrepEntry entry;
        entry.key = key;
        entry.localCbrNorm = stateValues[0];
        entry.neighborNorm = stateValues[1];
        entry.hopNorm = stateValues[2];
        entry.isOriginatorNear = stateValues[3];
        entry.lowThreshold = lowThreshold;
        entry.highThreshold = highThreshold;
        dlBucketBasedRrepEntriesByKey[key] = entry;
    }

    for (const auto& kv : dlBucketBasedRrepEntriesByKey)
        dlBucketBasedRrepEntries.push_back(kv.second);

    if (dlBucketBasedRrepEntries.empty())
        throw cRuntimeError("No valid entries were loaded from dlBucketBasedRrepLookupTable");
}

void Aodv::loadStateLookupBasedRrepParameters()
{
    stateLookupBasedRrepEntriesByKey.clear();
    stateLookupBasedRrepEntries.clear();

    int enabledPredictors = (dlBasedRrepEnabled ? 1 : 0) + (dlDirectThresholdRrepEnabled ? 1 : 0) + (dlBucketBasedRrepEnabled ? 1 : 0) + (stateLookupBasedRrepEnabled ? 1 : 0) + (treeBasedRrepEnabled ? 1 : 0);
    if (enabledPredictors > 1)
        throw cRuntimeError("Only one of dlBasedRrepEnabled, dlDirectThresholdRrepEnabled, dlBucketBasedRrepEnabled, stateLookupBasedRrepEnabled, or treeBasedRrepEnabled may be enabled at a time");

    if (!stateLookupBasedRrepEnabled)
        return;

    if (stateLookupBasedRrepCbrBinSize <= 0)
        throw cRuntimeError("stateLookupBasedRrepCbrBinSize must be positive");
    if (stateLookupBasedRrepNeighborBinSize <= 0)
        throw cRuntimeError("stateLookupBasedRrepNeighborBinSize must be positive");
    if (stateLookupBasedRrepHopMax <= 0)
        throw cRuntimeError("stateLookupBasedRrepHopMax must be positive");
    if (stateLookupBasedRrepPolicyCsvPath.empty())
        throw cRuntimeError("stateLookupBasedRrepPolicyCsvPath must not be empty when stateLookupBasedRrepEnabled is true");

    std::ifstream in(stateLookupBasedRrepPolicyCsvPath);
    if (!in.is_open())
        throw cRuntimeError("Cannot open stateLookupBasedRrepPolicyCsvPath '%s'", stateLookupBasedRrepPolicyCsvPath.c_str());

    std::string line;
    bool headerSkipped = false;
    while (std::getline(in, line)) {
        if (!headerSkipped) {
            headerSkipped = true;
            if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF && static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
                line = line.substr(3);
            if (line.find("stateCbrBin") != std::string::npos)
                continue;
        }

        std::stringstream rowStream(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(rowStream, cell, ','))
            cells.push_back(trimCopy(cell));

        if (cells.size() < 6)
            continue;

        StateLookupBasedRrepEntry entry;
        try {
            entry.stateCbrBin = std::stoi(cells[0]);
            entry.stateNeighborBin = std::stoi(cells[1]);
            entry.stateHopBin = std::stoi(cells[2]);
            entry.stateDirectBin = std::stoi(cells[3]);
            entry.lowThreshold = std::stod(cells[4]);
            entry.highThreshold = std::stod(cells[5]);
        }
        catch (const std::exception& e) {
            throw cRuntimeError("Invalid state lookup CSV row '%s': %s", line.c_str(), e.what());
        }

        if (!(entry.lowThreshold < entry.highThreshold))
            throw cRuntimeError("Invalid state lookup CSV row '%s': low threshold must be smaller than high threshold", line.c_str());

        entry.key = std::to_string(entry.stateCbrBin) + "|" +
            std::to_string(entry.stateNeighborBin) + "|" +
            std::to_string(entry.stateHopBin) + "|" +
            std::to_string(entry.stateDirectBin);
        stateLookupBasedRrepEntriesByKey[entry.key] = entry;
    }

    for (const auto& kv : stateLookupBasedRrepEntriesByKey)
        stateLookupBasedRrepEntries.push_back(kv.second);

    if (stateLookupBasedRrepEntries.empty())
        throw cRuntimeError("No valid entries were loaded from stateLookupBasedRrepPolicyCsvPath '%s'", stateLookupBasedRrepPolicyCsvPath.c_str());
}

void Aodv::loadTreeBasedRrepParameters()
{
    int enabledPredictors = (dlBasedRrepEnabled ? 1 : 0) + (dlDirectThresholdRrepEnabled ? 1 : 0) + (dlBucketBasedRrepEnabled ? 1 : 0) + (stateLookupBasedRrepEnabled ? 1 : 0) + (treeBasedRrepEnabled ? 1 : 0);
    if (enabledPredictors > 1)
        throw cRuntimeError("Only one of dlBasedRrepEnabled, dlDirectThresholdRrepEnabled, dlBucketBasedRrepEnabled, stateLookupBasedRrepEnabled, or treeBasedRrepEnabled may be enabled at a time");

    if (!treeBasedRrepEnabled)
        return;

    if (treeBasedRrepFeatureSet != "basic4" && treeBasedRrepFeatureSet != "local6" && treeBasedRrepFeatureSet != "local10")
        throw cRuntimeError("treeBasedRrepFeatureSet must be one of 'basic4', 'local6', or 'local10'");
    if (treeBasedRrepNeighborNorm <= 0)
        throw cRuntimeError("treeBasedRrepNeighborNorm must be positive");
    if (treeBasedRrepHopNorm <= 0)
        throw cRuntimeError("treeBasedRrepHopNorm must be positive");
    if (treeBasedRrepThresholdMin >= treeBasedRrepThresholdMax)
        throw cRuntimeError("treeBasedRrepThresholdMin must be smaller than treeBasedRrepThresholdMax");
    if (treeBasedRrepMinThresholdGap < 0)
        throw cRuntimeError("treeBasedRrepMinThresholdGap must be non-negative");
    if (treeBasedRrepMinThresholdGap > treeBasedRrepThresholdMax - treeBasedRrepThresholdMin)
        throw cRuntimeError("treeBasedRrepMinThresholdGap must not exceed the threshold range");
    if (treeBasedRrepLowModel.empty() || treeBasedRrepHighModel.empty())
        throw cRuntimeError("treeBasedRrepLowModel and treeBasedRrepHighModel must not be empty when treeBasedRrepEnabled is true");

    size_t inputSize = getRrepFeatureInputSize(treeBasedRrepFeatureSet);
    treeBasedRrepLowEnsemble = parseTreeBasedRrepEnsemble(treeBasedRrepLowModel, inputSize, "treeBasedRrepLowModel");
    treeBasedRrepHighEnsemble = parseTreeBasedRrepEnsemble(treeBasedRrepHighModel, inputSize, "treeBasedRrepHighModel");
}

std::pair<double, double> Aodv::inferDlBasedRrepThresholdRange(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    constexpr size_t outputSize = 2;
    size_t inputSize = getDlBasedRrepInputSize();
    size_t hidden1Size = dlBasedRrepCustomArchitectureEnabled ? static_cast<size_t>(dlBasedRrepHidden1Size) : 32;
    size_t hidden2Size = dlBasedRrepCustomArchitectureEnabled ? static_cast<size_t>(dlBasedRrepHidden2Size) : 16;
    size_t hidden3Size = dlBasedRrepCustomArchitectureEnabled ? static_cast<size_t>(dlBasedRrepHidden3Size) : 128;
    std::vector<double> inputs = buildDlBasedRrepInputs(localCbr, neighborCount, hopCount, isDirectRouteToDestination);

    std::vector<double> hidden1(hidden1Size, 0.0);
    for (size_t neuron = 0; neuron < hidden1.size(); ++neuron) {
        double sum = dlBasedRrepHiddenBiases[neuron];
        for (size_t feature = 0; feature < inputSize; ++feature)
            sum += dlBasedRrepHiddenWeights[neuron * inputSize + feature] * inputs[feature];
        hidden1[neuron] = relu(sum);
    }

    std::vector<double> hidden2(hidden2Size, 0.0);
    for (size_t neuron = 0; neuron < hidden2.size(); ++neuron) {
        double sum = dlBasedRrepHidden2Biases[neuron];
        for (size_t feature = 0; feature < hidden1Size; ++feature)
            sum += dlBasedRrepHidden2Weights[neuron * hidden1Size + feature] * hidden1[feature];
        hidden2[neuron] = relu(sum);
    }

    const std::vector<double> *outputInputs = &hidden2;
    size_t outputInputSize = hidden2Size;
    std::vector<double> hidden3;
    if (dlBasedRrepDirectThresholdOutputEnabled) {
        hidden3.assign(hidden3Size, 0.0);
        for (size_t neuron = 0; neuron < hidden3.size(); ++neuron) {
            double sum = dlBasedRrepHidden3Biases[neuron];
            for (size_t feature = 0; feature < hidden2Size; ++feature)
                sum += dlBasedRrepHidden3Weights[neuron * hidden2Size + feature] * hidden2[feature];
            hidden3[neuron] = relu(sum);
        }
        outputInputs = &hidden3;
        outputInputSize = hidden3Size;
    }

    std::array<double, outputSize> outputs = {};
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        double sum = dlBasedRrepOutputBiases[outputIndex];
        for (size_t neuron = 0; neuron < outputInputSize; ++neuron)
            sum += dlBasedRrepOutputWeights[outputIndex * outputInputSize + neuron] * (*outputInputs)[neuron];
        outputs[outputIndex] = sum;
    }

    if (dlBasedRrepDirectThresholdOutputEnabled) {
        double predictedLow = std::max(dlBasedRrepThresholdMin, std::min(dlBasedRrepThresholdMax, outputs[0]));
        double predictedHigh = std::max(dlBasedRrepThresholdMin, std::min(dlBasedRrepThresholdMax, outputs[1]));
        double minGap = std::min(dlBasedRrepMinThresholdGap, dlBasedRrepThresholdMax - dlBasedRrepThresholdMin);
        if (predictedHigh - predictedLow < minGap) {
            double center = (predictedLow + predictedHigh) / 2.0;
            double halfGap = minGap / 2.0;
            predictedLow = center - halfGap;
            predictedHigh = center + halfGap;
            if (predictedLow < dlBasedRrepThresholdMin) {
                predictedLow = dlBasedRrepThresholdMin;
                predictedHigh = dlBasedRrepThresholdMin + minGap;
            }
            if (predictedHigh > dlBasedRrepThresholdMax) {
                predictedHigh = dlBasedRrepThresholdMax;
                predictedLow = dlBasedRrepThresholdMax - minGap;
            }
        }
        return {predictedLow, predictedHigh};
    }

    double valueRange = dlBasedRrepThresholdMax - dlBasedRrepThresholdMin;
    double effectiveMinGap = std::min(dlBasedRrepMinThresholdGap, valueRange);
    double maxGapExtra = std::max(0.0, valueRange - effectiveMinGap);

    double centerNorm = sigmoid(outputs[0]);
    double gapExtra = std::min(maxGapExtra, softplus(outputs[1]));
    double predictedGap = effectiveMinGap + gapExtra;

    double centerMin = dlBasedRrepThresholdMin + predictedGap / 2.0;
    double centerMax = dlBasedRrepThresholdMax - predictedGap / 2.0;
    double centerSpan = std::max(0.0, centerMax - centerMin);
    double predictedCenter = centerMin + centerNorm * centerSpan;

    double predictedLow = predictedCenter - predictedGap / 2.0;
    double predictedHigh = predictedCenter + predictedGap / 2.0;

    return {predictedLow, predictedHigh};
}

std::pair<double, double> Aodv::inferDlDirectThresholdRrepThresholdRange(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination, std::vector<double> *debugInputs, std::array<double, 2> *debugRawOutputs) const
{
    constexpr size_t outputSize = 2;
    size_t inputSize = getRrepFeatureInputSize(dlDirectThresholdRrepFeatureSet);
    size_t hidden1Size = static_cast<size_t>(dlDirectThresholdRrepHidden1Size);
    size_t hidden2Size = static_cast<size_t>(dlDirectThresholdRrepHidden2Size);
    size_t hidden3Size = static_cast<size_t>(dlDirectThresholdRrepHidden3Size);
    std::vector<double> inputs = buildRrepFeatureInputs(dlDirectThresholdRrepFeatureSet, dlDirectThresholdRrepNeighborNorm, dlDirectThresholdRrepHopNorm, localCbr, neighborCount, hopCount, isDirectRouteToDestination);
    if (dlDirectThresholdRrepInputStandardizationEnabled) {
        for (size_t i = 0; i < inputs.size(); ++i)
            inputs[i] = (inputs[i] - dlDirectThresholdRrepInputMeans[i]) / dlDirectThresholdRrepInputScales[i];
    }
    if (debugInputs != nullptr)
        *debugInputs = inputs;

    std::vector<double> hidden1(hidden1Size, 0.0);
    for (size_t neuron = 0; neuron < hidden1.size(); ++neuron) {
        double sum = dlDirectThresholdRrepHiddenBiases[neuron];
        for (size_t feature = 0; feature < inputSize; ++feature)
            sum += dlDirectThresholdRrepHiddenWeights[neuron * inputSize + feature] * inputs[feature];
        hidden1[neuron] = relu(sum);
    }

    std::vector<double> hidden2(hidden2Size, 0.0);
    for (size_t neuron = 0; neuron < hidden2.size(); ++neuron) {
        double sum = dlDirectThresholdRrepHidden2Biases[neuron];
        for (size_t feature = 0; feature < hidden1Size; ++feature)
            sum += dlDirectThresholdRrepHidden2Weights[neuron * hidden1Size + feature] * hidden1[feature];
        hidden2[neuron] = relu(sum);
    }

    std::vector<double> hidden3(hidden3Size, 0.0);
    for (size_t neuron = 0; neuron < hidden3.size(); ++neuron) {
        double sum = dlDirectThresholdRrepHidden3Biases[neuron];
        for (size_t feature = 0; feature < hidden2Size; ++feature)
            sum += dlDirectThresholdRrepHidden3Weights[neuron * hidden2Size + feature] * hidden2[feature];
        hidden3[neuron] = relu(sum);
    }

    std::array<double, outputSize> outputs = {};
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        double sum = dlDirectThresholdRrepOutputBiases[outputIndex];
        for (size_t neuron = 0; neuron < hidden3Size; ++neuron)
            sum += dlDirectThresholdRrepOutputWeights[outputIndex * hidden3Size + neuron] * hidden3[neuron];
        outputs[outputIndex] = sum;
    }
    if (dlDirectThresholdRrepOutputStandardizationEnabled) {
        for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex)
            outputs[outputIndex] = outputs[outputIndex] * dlDirectThresholdRrepOutputScales[outputIndex] + dlDirectThresholdRrepOutputMeans[outputIndex];
    }
    if (debugRawOutputs != nullptr)
        *debugRawOutputs = outputs;

    double predictedLow = std::max(dlDirectThresholdRrepThresholdMin, std::min(dlDirectThresholdRrepThresholdMax, outputs[0]));
    double predictedHigh = std::max(dlDirectThresholdRrepThresholdMin, std::min(dlDirectThresholdRrepThresholdMax, outputs[1]));
    double minGap = std::min(dlDirectThresholdRrepMinThresholdGap, dlDirectThresholdRrepThresholdMax - dlDirectThresholdRrepThresholdMin);
    if (predictedHigh - predictedLow < minGap) {
        double center = (predictedLow + predictedHigh) / 2.0;
        double halfGap = minGap / 2.0;
        predictedLow = center - halfGap;
        predictedHigh = center + halfGap;
        if (predictedLow < dlDirectThresholdRrepThresholdMin) {
            predictedLow = dlDirectThresholdRrepThresholdMin;
            predictedHigh = dlDirectThresholdRrepThresholdMin + minGap;
        }
        if (predictedHigh > dlDirectThresholdRrepThresholdMax) {
            predictedHigh = dlDirectThresholdRrepThresholdMax;
            predictedLow = dlDirectThresholdRrepThresholdMax - minGap;
        }
    }

    return {predictedLow, predictedHigh};
}

std::string Aodv::buildDlBucketBasedStateKey(double localCbrNorm, double neighborNorm, double hopNorm, double isOriginatorNear) const
{
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(dlBucketBasedRrepBucketRoundDigits);
    os << roundToDigits(clamp01(localCbrNorm), dlBucketBasedRrepBucketRoundDigits) << "|"
       << roundToDigits(clamp01(neighborNorm), dlBucketBasedRrepBucketRoundDigits) << "|"
       << roundToDigits(clamp01(hopNorm), dlBucketBasedRrepBucketRoundDigits) << "|"
       << roundToDigits(clamp01(isOriginatorNear), dlBucketBasedRrepBucketRoundDigits);
    return os.str();
}

std::pair<double, double> Aodv::inferDlBucketBasedRrepThresholdRange(double localCbr, int neighborCount, unsigned int rreqHopCount) const
{
    double localCbrNorm = clamp01(localCbr / 100.0);
    double neighborNorm = clamp01(static_cast<double>(neighborCount) / dlBucketBasedRrepNeighborNorm);
    double hopNorm = clamp01(static_cast<double>(rreqHopCount) / dlBucketBasedRrepHopNorm);
    double isOriginatorNear = rreqHopCount <= 1 ? 1.0 : 0.0;

    std::string stateKey = buildDlBucketBasedStateKey(localCbrNorm, neighborNorm, hopNorm, isOriginatorNear);
    auto exactIt = dlBucketBasedRrepEntriesByKey.find(stateKey);
    if (exactIt != dlBucketBasedRrepEntriesByKey.end())
        return {exactIt->second.lowThreshold, exactIt->second.highThreshold};

    if (!dlBucketBasedRrepNearestFallbackEnabled || dlBucketBasedRrepEntries.empty())
        throw cRuntimeError("No bucket prediction found for key '%s' and nearest fallback is disabled", stateKey.c_str());

    const DlBucketBasedRrepEntry *bestEntry = nullptr;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const auto& entry : dlBucketBasedRrepEntries) {
        double distance =
            (entry.localCbrNorm - localCbrNorm) * (entry.localCbrNorm - localCbrNorm) +
            (entry.neighborNorm - neighborNorm) * (entry.neighborNorm - neighborNorm) +
            (entry.hopNorm - hopNorm) * (entry.hopNorm - hopNorm) +
            (entry.isOriginatorNear - isOriginatorNear) * (entry.isOriginatorNear - isOriginatorNear);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestEntry = &entry;
        }
    }

    if (bestEntry == nullptr)
        throw cRuntimeError("Failed to resolve nearest bucket prediction for key '%s'", stateKey.c_str());

    return {bestEntry->lowThreshold, bestEntry->highThreshold};
}

std::string Aodv::buildStateLookupBasedRrepStateKey(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    int stateCbrBin = static_cast<int>(std::floor(std::max(0.0, localCbr) / stateLookupBasedRrepCbrBinSize)) * stateLookupBasedRrepCbrBinSize;
    stateCbrBin = std::max(0, std::min(100, stateCbrBin));
    int stateNeighborBin = static_cast<int>(std::floor(std::max(0, neighborCount) / static_cast<double>(stateLookupBasedRrepNeighborBinSize))) * stateLookupBasedRrepNeighborBinSize;
    stateNeighborBin = std::max(0, std::min(100, stateNeighborBin));
    int stateHopBin = static_cast<int>(std::lround(static_cast<double>(hopCount)));
    stateHopBin = std::max(0, std::min(stateLookupBasedRrepHopMax, stateHopBin));
    int stateDirectBin = isDirectRouteToDestination ? 1 : 0;
    return std::to_string(stateCbrBin) + "|" +
        std::to_string(stateNeighborBin) + "|" +
        std::to_string(stateHopBin) + "|" +
        std::to_string(stateDirectBin);
}

std::pair<double, double> Aodv::inferStateLookupBasedRrepThresholdRange(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    std::string stateKey = buildStateLookupBasedRrepStateKey(localCbr, neighborCount, hopCount, isDirectRouteToDestination);
    auto exactIt = stateLookupBasedRrepEntriesByKey.find(stateKey);
    if (exactIt != stateLookupBasedRrepEntriesByKey.end())
        return {exactIt->second.lowThreshold, exactIt->second.highThreshold};

    if (!stateLookupBasedRrepNearestFallbackEnabled || stateLookupBasedRrepEntries.empty())
        throw cRuntimeError("No state lookup prediction found for key '%s' and nearest fallback is disabled", stateKey.c_str());

    int targetCbrBin = static_cast<int>(std::floor(std::max(0.0, localCbr) / stateLookupBasedRrepCbrBinSize)) * stateLookupBasedRrepCbrBinSize;
    targetCbrBin = std::max(0, std::min(100, targetCbrBin));
    int targetNeighborBin = static_cast<int>(std::floor(std::max(0, neighborCount) / static_cast<double>(stateLookupBasedRrepNeighborBinSize))) * stateLookupBasedRrepNeighborBinSize;
    targetNeighborBin = std::max(0, std::min(100, targetNeighborBin));
    int targetHopBin = static_cast<int>(std::lround(static_cast<double>(hopCount)));
    targetHopBin = std::max(0, std::min(stateLookupBasedRrepHopMax, targetHopBin));
    int targetDirectBin = isDirectRouteToDestination ? 1 : 0;

    const StateLookupBasedRrepEntry *bestEntry = nullptr;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const auto& entry : stateLookupBasedRrepEntries) {
        double distance =
            std::abs(entry.stateCbrBin - targetCbrBin) / static_cast<double>(stateLookupBasedRrepCbrBinSize) +
            std::abs(entry.stateNeighborBin - targetNeighborBin) / static_cast<double>(stateLookupBasedRrepNeighborBinSize) +
            std::abs(entry.stateHopBin - targetHopBin) +
            2.0 * std::abs(entry.stateDirectBin - targetDirectBin);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestEntry = &entry;
        }
    }

    if (bestEntry == nullptr)
        throw cRuntimeError("Failed to resolve nearest state lookup prediction for key '%s'", stateKey.c_str());

    return {bestEntry->lowThreshold, bestEntry->highThreshold};
}

std::pair<double, double> Aodv::inferTreeBasedRrepThresholdRange(double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination) const
{
    std::vector<double> inputs = buildRrepFeatureInputs(treeBasedRrepFeatureSet, treeBasedRrepNeighborNorm, treeBasedRrepHopNorm, localCbr, neighborCount, hopCount, isDirectRouteToDestination);
    double predictedLow = evaluateTreeBasedRrepEnsemble(treeBasedRrepLowEnsemble, inputs, "treeBasedRrepLowModel");
    double predictedHigh = evaluateTreeBasedRrepEnsemble(treeBasedRrepHighEnsemble, inputs, "treeBasedRrepHighModel");

    predictedLow = std::max(treeBasedRrepThresholdMin, std::min(treeBasedRrepThresholdMax, predictedLow));
    predictedHigh = std::max(treeBasedRrepThresholdMin, std::min(treeBasedRrepThresholdMax, predictedHigh));

    double minGap = std::min(treeBasedRrepMinThresholdGap, treeBasedRrepThresholdMax - treeBasedRrepThresholdMin);
    if (predictedHigh - predictedLow < minGap) {
        double center = (predictedLow + predictedHigh) / 2.0;
        double halfGap = minGap / 2.0;
        predictedLow = center - halfGap;
        predictedHigh = center + halfGap;
        if (predictedLow < treeBasedRrepThresholdMin) {
            predictedLow = treeBasedRrepThresholdMin;
            predictedHigh = treeBasedRrepThresholdMin + minGap;
        }
        if (predictedHigh > treeBasedRrepThresholdMax) {
            predictedHigh = treeBasedRrepThresholdMax;
            predictedLow = treeBasedRrepThresholdMax - minGap;
        }
    }

    return {predictedLow, predictedHigh};
}

const Ptr<Rreq> Aodv::createRREQ(const L3Address& destAddr)
{
    auto rreqPacket = makeShared<Rreq>(); // TODO "AODV-RREQ");
    rreqPacket->setPacketType(usingIpv6 ? RREQ_IPv6 : RREQ);
    rreqPacket->setChunkLength(usingIpv6 ? B(48) : B(24));

    rreqPacket->setGratuitousRREPFlag(askGratuitousRREP);
    IRoute *lastKnownRoute = routingTable->findBestMatchingRoute(destAddr);

    // The Originator Sequence Number in the RREQ message is the
    // node's own sequence number, which is incremented prior to
    // insertion in a RREQ.
    sequenceNum++;

    rreqPacket->setOriginatorSeqNum(sequenceNum);

    if (lastKnownRoute && lastKnownRoute->getSource() == this) {
        // The Destination Sequence Number field in the RREQ message is the last
        // known destination sequence number for this destination and is copied
        // from the Destination Sequence Number field in the routing table.

        AodvRouteData *routeData = check_and_cast<AodvRouteData *>(lastKnownRoute->getProtocolData());
        if (routeData && routeData->hasValidDestNum()) {
            rreqPacket->setDestSeqNum(routeData->getDestSeqNum());
            rreqPacket->setUnknownSeqNumFlag(false);
        }
        else
            rreqPacket->setUnknownSeqNumFlag(true);
    }
    else
        rreqPacket->setUnknownSeqNumFlag(true); // If no sequence number is known, the unknown sequence number flag MUST be set.

    rreqPacket->setOriginatorAddr(getSelfIPAddress());
    rreqPacket->setDestAddr(destAddr);

    // The RREQ ID field is incremented by one from the last RREQ ID used
    // by the current node. Each node maintains only one RREQ ID.
    rreqId++;
    rreqPacket->setRreqId(rreqId);

    // The Hop Count field is set to zero.
    rreqPacket->setHopCount(0);

    // Destination only flag (D) indicates that only the
    // destination may respond to this RREQ.
    rreqPacket->setDestOnlyFlag(destinationOnlyFlag);

    // Before broadcasting the RREQ, the originating node buffers the RREQ
    // ID and the Originator IP address (its own address) of the RREQ for
    // PATH_DISCOVERY_TIME.
    // In this way, when the node receives the packet again from its neighbors,
    // it will not reprocess and re-forward the packet.

    RreqIdentifier rreqIdentifier(getSelfIPAddress(), rreqId);
    rreqsArrivalTime[rreqIdentifier] = simTime();
    return rreqPacket;
}

const Ptr<Rrep> Aodv::createRREP(const Ptr<Rreq>& rreq, IRoute *destRoute, IRoute *originatorRoute, const L3Address& lastHopAddr)
{
    auto rrep = makeShared<Rrep>(); // TODO "AODV-RREP");
    rrep->setPacketType(usingIpv6 ? RREP_IPv6 : RREP);
    rrep->setChunkLength(usingIpv6 ? B(44) : B(20));

    // When generating a RREP message, a node copies the Destination IP
    // Address and the Originator Sequence Number from the RREQ message into
    // the corresponding fields in the RREP message.

    rrep->setDestAddr(rreq->getDestAddr());

    // OriginatorAddr = The IP address of the node which originated the RREQ
    // for which the route is supplied.
    rrep->setOriginatorAddr(rreq->getOriginatorAddr());

    // Processing is slightly different, depending on whether the node is
    // itself the requested destination (see section 6.6.1), or instead
    // if it is an intermediate node with an fresh enough route to the destination
    // (see section 6.6.2).

    if (rreq->getDestAddr() == getSelfIPAddress()) { // node is itself the requested destination
        // 6.6.1. Route Reply Generation by the Destination

        // If the generating node is the destination itself, it MUST increment
        // its own sequence number by one if the sequence number in the RREQ
        // packet is equal to that incremented value.

        if (!rreq->getUnknownSeqNumFlag() && sequenceNum + 1 == rreq->getDestSeqNum())
            sequenceNum++;

        // The destination node places its (perhaps newly incremented)
        // sequence number into the Destination Sequence Number field of
        // the RREP,
        rrep->setDestSeqNum(sequenceNum);

        // and enters the value zero in the Hop Count field
        // of the RREP.
        rrep->setHopCount(0);

        // The destination node copies the value MY_ROUTE_TIMEOUT
        // into the Lifetime field of the RREP.
        rrep->setLifeTime(myRouteTimeout.trunc(SIMTIME_MS));
    }
    else { // intermediate node
        // 6.6.2. Route Reply Generation by an Intermediate Node
        if (destRoute == nullptr || originatorRoute == nullptr) {
            EV_WARN << "Cannot build intermediate RREP because destination/originator route is missing. "
                    << "destRoute=" << (destRoute != nullptr)
                    << ", originatorRoute=" << (originatorRoute != nullptr) << endl;
            rrep->setDestSeqNum(rreq->getDestSeqNum());
            rrep->setHopCount(rreq->getHopCount());
            rrep->setLifeTime(activeRouteTimeout.trunc(SIMTIME_MS));
            return rrep;
        }

        // it copies its known sequence number for the destination into
        // the Destination Sequence Number field in the RREP message.
        AodvRouteData *destRouteData = dynamic_cast<AodvRouteData *>(destRoute->getProtocolData());
        AodvRouteData *originatorRouteData = dynamic_cast<AodvRouteData *>(originatorRoute->getProtocolData());
        if (destRouteData == nullptr || originatorRouteData == nullptr) {
            EV_WARN << "Cannot build intermediate RREP because route protocol data is missing. "
                    << "destRouteData=" << (destRouteData != nullptr)
                    << ", originatorRouteData=" << (originatorRouteData != nullptr) << endl;
            rrep->setDestSeqNum(rreq->getDestSeqNum());
            rrep->setHopCount(rreq->getHopCount());
            rrep->setLifeTime(activeRouteTimeout.trunc(SIMTIME_MS));
            return rrep;
        }
        rrep->setDestSeqNum(destRouteData->getDestSeqNum());

        // The intermediate node updates the forward route entry by placing the
        // last hop node (from which it received the RREQ, as indicated by the
        // source IP address field in the IP header) into the precursor list for
        // the forward route entry -- i.e., the entry for the Destination IP
        // Address.
        destRouteData->addPrecursor(lastHopAddr);
        logPrecursorAddition("HANDLE_RREP_DEST", destRoute->getDestinationAsGeneric(), lastHopAddr, destRouteData->getPrecursorList());

        // The intermediate node also updates its route table entry
        // for the node originating the RREQ by placing the next hop towards the
        // destination in the precursor list for the reverse route entry --
        // i.e., the entry for the Originator IP Address field of the RREQ
        // message data.

        originatorRouteData->addPrecursor(destRoute->getNextHopAsGeneric());
        logPrecursorAddition("HANDLE_RREP_ORIGINATOR", originatorRoute->getDestinationAsGeneric(), destRoute->getNextHopAsGeneric(), originatorRouteData->getPrecursorList());

        // The intermediate node places its distance in hops from the
        // destination (indicated by the hop count in the routing table)
        // Hop Count field in the RREP.

        rrep->setHopCount(destRoute->getMetric());

        // The Lifetime field of the RREP is calculated by subtracting the
        // current time from the expiration time in its route table entry.

        rrep->setLifeTime((destRouteData->getLifeTime() - simTime()).trunc(SIMTIME_MS));
    }

    return rrep;
}

const Ptr<Rrep> Aodv::createGratuitousRREP(const Ptr<Rreq>& rreq, IRoute *originatorRoute)
{
    ASSERT(originatorRoute != nullptr);
    auto grrep = makeShared<Rrep>(); // TODO "AODV-GRREP");
    grrep->setPacketType(usingIpv6 ? RREP_IPv6 : RREP);
    grrep->setChunkLength(usingIpv6 ? B(44) : B(20));

    AodvRouteData *routeData = dynamic_cast<AodvRouteData *>(originatorRoute->getProtocolData());
    if (routeData == nullptr) {
        EV_WARN << "Cannot build gratuitous RREP because originator route protocol data is missing" << endl;
        grrep->setHopCount(originatorRoute->getMetric());
        grrep->setDestAddr(rreq->getOriginatorAddr());
        grrep->setDestSeqNum(rreq->getOriginatorSeqNum());
        grrep->setOriginatorAddr(rreq->getDestAddr());
        grrep->setLifeTime(activeRouteTimeout.trunc(SIMTIME_MS));
        return grrep;
    }

    // Hop Count                        The Hop Count as indicated in the
    //                                  node's route table entry for the
    //                                  originator
    //
    // Destination IP Address           The IP address of the node that
    //                                  originated the RREQ
    //
    // Destination Sequence Number      The Originator Sequence Number from
    //                                  the RREQ
    //
    // Originator IP Address            The IP address of the Destination
    //                                  node in the RREQ
    //
    // Lifetime                         The remaining lifetime of the route
    //                                  towards the originator of the RREQ,
    //                                  as known by the intermediate node.

    grrep->setHopCount(originatorRoute->getMetric());
    grrep->setDestAddr(rreq->getOriginatorAddr());
    grrep->setDestSeqNum(rreq->getOriginatorSeqNum());
    grrep->setOriginatorAddr(rreq->getDestAddr());
    grrep->setLifeTime(routeData->getLifeTime());
    return grrep;
}

void Aodv::handleRREP(const Ptr<Rrep>& rrep, const L3Address& sourceAddr)
{
    // 6.7. Receiving and Forwarding Route Replies

    EV_INFO << "AODV Route Reply arrived with source addr: " << sourceAddr << " originator addr: " << rrep->getOriginatorAddr()
                    << " destination addr: " << rrep->getDestAddr() << endl;
    /*
    appendAodvMetric("aodv_control_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RREP_RECV, source=" + sourceAddr.str() +
            ", originator=" + rrep->getOriginatorAddr().str() +
            ", destination=" + rrep->getDestAddr().str() +
            ", hopCount=" + std::to_string(rrep->getHopCount()));*/

    if (rrep->getOriginatorAddr().isUnspecified()) {
        EV_INFO << "This Route Reply is a Hello Message" << endl;
        handleHelloMessage(rrep);
        return;
    }
    if (cbrRrepMetricsEnabled && hasOngoingRouteDiscovery(rrep->getDestAddr())) {
        metricsRrepReceivedCount++;
        metricsRouteDiscoveryCandidateCounts[rrep->getDestAddr()]++;
    }
    // When a node receives a RREP message, it searches (using longest-
    // prefix matching) for a route to the previous hop.

    // If needed, a route is created for the previous hop,
    // but without a valid sequence number (see section 6.2)

    IRoute *previousHopRoute = routingTable->findBestMatchingRoute(sourceAddr);

    if (!previousHopRoute || previousHopRoute->getSource() != this) {
        // create without valid sequence number
        previousHopRoute = createRoute(sourceAddr, sourceAddr, 1, false, rrep->getDestSeqNum(), true, simTime() + activeRouteTimeout);
    }
    else
        updateRoutingTable(previousHopRoute, sourceAddr, 1, false, rrep->getDestSeqNum(), true, simTime() + activeRouteTimeout);

    // Next, the node then increments the hop count value in the RREP by one,
    // to account for the new hop through the intermediate node
    unsigned int newHopCount = rrep->getHopCount() + 1;
    rrep->setHopCount(newHopCount);

    // Then the forward route for this destination is created if it does not
    // already exist.

    IRoute *destRoute = routingTable->findBestMatchingRoute(rrep->getDestAddr());
    AodvRouteData *destRouteData = nullptr;
    simtime_t lifeTime = rrep->getLifeTime();
    unsigned int destSeqNum = rrep->getDestSeqNum();

    if (destRoute && destRoute->getSource() == this) { // already exists
        destRouteData = dynamic_cast<AodvRouteData *>(destRoute->getProtocolData());
        if (destRouteData == nullptr) {
            EV_WARN << "Dropping RREP processing because destination route protocol data is missing for "
                    << rrep->getDestAddr() << endl;
            return;
        }
        // Upon comparison, the existing entry is updated only in the following circumstances:

        // (i) the sequence number in the routing table is marked as
        // invalid in route table entry.

        if (!destRouteData->hasValidDestNum()) {
            updateRoutingTable(destRoute, sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);

            // If the route table entry to the destination is created or updated,
            // then the following actions occur:
            //
            // -  the route is marked as active,
            //
            // -  the destination sequence number is marked as valid,
            //
            // -  the next hop in the route entry is assigned to be the node from
            //    which the RREP is received, which is indicated by the source IP
            //    address field in the IP header,
            //
            // -  the hop count is set to the value of the New Hop Count,
            //
            // -  the expiry time is set to the current time plus the value of the
            //    Lifetime in the RREP message,
            //
            // -  and the destination sequence number is the Destination Sequence
            //    Number in the RREP message.
        }
        // (ii) the Destination Sequence Number in the RREP is greater than
        //      the node's copy of the destination sequence number and the
        //      known value is valid, or
        else if (destSeqNum > destRouteData->getDestSeqNum()) {
            updateRoutingTable(destRoute, sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);
        }
        else {
            // (iii) the sequence numbers are the same, but the route is
            //       marked as inactive, or
            if (destSeqNum == destRouteData->getDestSeqNum() && !destRouteData->isActive()) {
                updateRoutingTable(destRoute, sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);
            }
            // (iv) the sequence numbers are the same, and the New Hop Count is
            //      smaller than the hop count in route table entry.
            else if (destSeqNum == destRouteData->getDestSeqNum() && newHopCount < (unsigned int)destRoute->getMetric()) {
                updateRoutingTable(destRoute, sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);
            }
        }
    }
    else { // create forward route for the destination: this path will be used by the originator to send data packets
        destRoute = createRoute(rrep->getDestAddr(), sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);
        destRouteData = dynamic_cast<AodvRouteData *>(destRoute->getProtocolData());
        if (destRouteData == nullptr) {
            EV_WARN << "Dropping RREP processing because created destination route protocol data is missing for "
                    << rrep->getDestAddr() << endl;
            return;
        }
    }

    // If the current node is not the node indicated by the Originator IP
    // Address in the RREP message AND a forward route has been created or
    // updated as described above, the node consults its route table entry
    // for the originating node to determine the next hop for the RREP
    // packet, and then forwards the RREP towards the originator using the
    // information in that route table entry.

    IRoute *originatorRoute = routingTable->findBestMatchingRoute(rrep->getOriginatorAddr());
    if (getSelfIPAddress() != rrep->getOriginatorAddr()) {
        // If a node forwards a RREP over a link that is likely to have errors or
        // be unidirectional, the node SHOULD set the 'A' flag to require that the
        // recipient of the RREP acknowledge receipt of the RREP by sending a RREP-ACK
        // message back (see section 6.8).

        if (originatorRoute && originatorRoute->getSource() == this) {
            AodvRouteData *originatorRouteData = dynamic_cast<AodvRouteData *>(originatorRoute->getProtocolData());
            if (destRoute == nullptr || destRouteData == nullptr || originatorRouteData == nullptr) {
                EV_WARN << "Dropping RREP forward because one of the required routes/protocol data is missing. "
                        << "destRoute=" << (destRoute != nullptr)
                        << ", destRouteData=" << (destRouteData != nullptr)
                        << ", originatorRouteData=" << (originatorRouteData != nullptr) << endl;
                return;
            }

            // Also, at each node the (reverse) route used to forward a
            // RREP has its lifetime changed to be the maximum of (existing-
            // lifetime, (current time + ACTIVE_ROUTE_TIMEOUT).

            simtime_t existingLifeTime = originatorRouteData->getLifeTime();
            originatorRouteData->setLifeTime(std::max(simTime() + activeRouteTimeout, existingLifeTime));

            if (simTime() > rebootTime + deletePeriod || rebootTime == 0) {
                // If a node forwards a RREP over a link that is likely to have errors
                // or be unidirectional, the node SHOULD set the 'A' flag to require that
                // the recipient of the RREP acknowledge receipt of the RREP by sending a
                // RREP-ACK message back (see section 6.8).

                if (rrep->getAckRequiredFlag()) {
                    auto rrepACK = createRREPACK();
                    sendRREPACK(rrepACK, sourceAddr);
                    rrep->setAckRequiredFlag(false);
                }

                // When any node transmits a RREP, the precursor list for the
                // corresponding destination node is updated by adding to it
                // the next hop node to which the RREP is forwarded.

                destRouteData->addPrecursor(originatorRoute->getNextHopAsGeneric());
                logPrecursorAddition("FORWARD_RREP_DEST", destRoute->getDestinationAsGeneric(), originatorRoute->getNextHopAsGeneric(), destRouteData->getPrecursorList());

                // Finally, the precursor list for the next hop towards the
                // destination is updated to contain the next hop towards the
                // source (originator).

                IRoute *nextHopToDestRoute = routingTable->findBestMatchingRoute(destRoute->getNextHopAsGeneric());
                if (nextHopToDestRoute && nextHopToDestRoute->getSource() == this) {
                    AodvRouteData *nextHopToDestRouteData = dynamic_cast<AodvRouteData *>(nextHopToDestRoute->getProtocolData());
                    if (nextHopToDestRouteData != nullptr) {
                        nextHopToDestRouteData->addPrecursor(originatorRoute->getNextHopAsGeneric());
                        logPrecursorAddition("FORWARD_RREP_NEXT_HOP", nextHopToDestRoute->getDestinationAsGeneric(), originatorRoute->getNextHopAsGeneric(), nextHopToDestRouteData->getPrecursorList());
                    }
                    else {
                        EV_WARN << "Skipping next-hop precursor update because route protocol data is missing for "
                                << nextHopToDestRoute->getDestinationAsGeneric() << endl;
                    }
                }
                auto outgoingRREP = dynamicPtrCast<Rrep>(rrep->dupShared());
                forwardRREP(outgoingRREP, originatorRoute->getNextHopAsGeneric(), 100);
            }
        }
        else
            EV_ERROR << "Reverse route doesn't exist. Dropping the RREP message" << endl;
    }
    else {
        if (hasOngoingRouteDiscovery(rrep->getDestAddr())) {
            EV_INFO << "The Route Reply has arrived for our Route Request to node " << rrep->getDestAddr() << endl;
            updateRoutingTable(destRoute, sourceAddr, newHopCount, true, destSeqNum, true, simTime() + lifeTime);
            completeRouteDiscovery(rrep->getDestAddr());
        }
    }
}

void Aodv::updateRoutingTable(IRoute *route, const L3Address& nextHop, unsigned int hopCount, bool hasValidDestNum, unsigned int destSeqNum, bool isActive, simtime_t lifeTime)
{
    EV_DETAIL << "Updating existing route: " << route << endl;

    route->setNextHop(nextHop);
    route->setMetric(hopCount);

    AodvRouteData *routingData = check_and_cast<AodvRouteData *>(route->getProtocolData());
    ASSERT(routingData != nullptr);

    routingData->setLifeTime(lifeTime);
    routingData->setDestSeqNum(destSeqNum);
    routingData->setIsActive(isActive);
    routingData->setHasValidDestNum(hasValidDestNum);
    logRouteGraphEvent("ROUTE_UPDATE", route->getDestinationAsGeneric(), nextHop, hopCount, isActive, lifeTime);
    logRouteCauseEvent("ROUTE_UPDATE", route->getDestinationAsGeneric(), nextHop, hopCount, isActive, lifeTime, "updateRoutingTable");
    logRoutingTableSnapshot("ROUTE_UPDATE");

    EV_DETAIL << "Route updated: " << route << endl;

    scheduleExpungeRoutes();
}

void Aodv::sendAODVPacket(const Ptr<AodvControlPacket>& aodvPacket, const L3Address& destAddr, unsigned int timeToLive, double delay)
{
    ASSERT(timeToLive != 0);

    std::string packetName;
    switch (aodvPacket->getPacketType()) {
        case RREQ:
        case RREQ_IPv6:
            packetName = "aodv::Rreq";
            break;
        case RERR:
        case RERR_IPv6:
            packetName = "aodv::Rerr";
            break;
        case RREP:
        case RREP_IPv6: {
            auto rrep = dynamicPtrCast<const Rrep>(aodvPacket);
            bool isHelloMessage = destAddr.isBroadcast() && rrep != nullptr && rrep->getDestAddr() == getSelfIPAddress() && rrep->getHopCount() == 0;
            packetName = isHelloMessage ? "aodv::Hello" : "aodv::Rrep";
            break;
        }
        default: {
            const char *className = aodvPacket->getClassName();
            packetName = !strncmp("inet::", className, 6) ? className + 6 : className;
            break;
        }
    }
    Packet *packet = new Packet(packetName.c_str(), aodvPacket);

    int interfaceId = CHK(interfaceTable->findInterfaceByName(par("interface")))->getInterfaceId(); // TODO Implement: support for multiple interfaces
    packet->addTag<InterfaceReq>()->setInterfaceId(interfaceId);
    packet->addTag<HopLimitReq>()->setHopLimit(timeToLive);
    packet->addTag<L3AddressReq>()->setDestAddress(destAddr);
    packet->addTag<L4PortReq>()->setDestPort(aodvUDPPort);

    if (destAddr.isBroadcast())
        lastBroadcastTime = simTime();

    if (delay == 0)
        socket.send(packet);
    else {
        auto *timer = new PacketHolderMessage("aodv-send-jitter", KIND_DELAYEDSEND);
        timer->setOwnedPacket(packet);
        scheduleAfter(delay, timer);
    }
}

void Aodv::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    // process incoming packet
    processPacket(packet);
}

void Aodv::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Ignoring UDP error report " << indication->getName() << endl;
    delete indication;
}

void Aodv::socketClosed(UdpSocket *socket)
{
    if (operationalState == State::STOPPING_OPERATION)
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
}

double Aodv::getLocalCbr() const
{
    cModule *radioModule = nullptr;
    if (host != nullptr) {
        cModule *wlanModule = host->getSubmodule("wlan", 0);
        if (wlanModule != nullptr)
            radioModule = wlanModule->getSubmodule("radio");
    }
    if (radioModule == nullptr)
        return 0.0;

    auto *radio = dynamic_cast<physicallayer::Radio *>(radioModule);
    if (radio == nullptr)
        return 0.0;

    return 100.0 * radio->getCurrentCbr();
}

int Aodv::countCurrentNeighbors() const
{
    if (useBdStationCount)
        return getBdStationCount();

    if (host == nullptr)
        return 0;
    cModule *mobilityModule = host->getSubmodule("mobility");
    if (mobilityModule == nullptr)
        return 0;
    auto *selfMobility = dynamic_cast<IMobility *>(mobilityModule);
    if (selfMobility == nullptr)
        return 0;

    double neighborRange = 200.0;
    cModule *network = host->getParentModule();
    if (network != nullptr) {
        cModule *connectionManager = network->getSubmodule("connectionManager");
        if (connectionManager != nullptr && connectionManager->hasPar("maxInterfDist"))
            neighborRange = connectionManager->par("maxInterfDist").doubleValue();
    }
    if (network == nullptr)
        return 0;

    Coord selfPosition = selfMobility->getCurrentPosition();
    int neighborCount = 0;
    for (const char *vectorName : {"node", "rsu"}) {
        for (int i = 0;; ++i) {
            cModule *candidate = network->getSubmodule(vectorName, i);
            if (candidate == nullptr)
                break;
            if (candidate == host)
                continue;
            cModule *candidateMobilityModule = candidate->getSubmodule("mobility");
            if (candidateMobilityModule == nullptr)
                continue;
            auto *candidateMobility = dynamic_cast<IMobility *>(candidateMobilityModule);
            if (candidateMobility == nullptr)
                continue;
            if (selfPosition.distance(candidateMobility->getCurrentPosition()) <= neighborRange)
                neighborCount++;
        }
    }
    return neighborCount;
}

int Aodv::getBdStationCount() const
{
    auto *networkInterface = interfaceTable->findInterfaceByName(par("interface"));
    if (networkInterface == nullptr)
        return 0;

    auto *interfaceModule = getSimulation()->findModuleByPath(networkInterface->getInterfaceFullPath().c_str());
    if (interfaceModule == nullptr && host != nullptr) {
        std::string interfaceName = networkInterface->getInterfaceName();
        size_t digitStart = interfaceName.find_first_of("0123456789");
        if (digitStart != std::string::npos) {
            std::string moduleName = interfaceName.substr(0, digitStart);
            int moduleIndex = std::stoi(interfaceName.substr(digitStart));
            interfaceModule = host->getSubmodule(moduleName.c_str(), moduleIndex);
        }
        if (interfaceModule == nullptr)
            interfaceModule = host->getSubmodule(interfaceName.c_str());
    }
    if (interfaceModule == nullptr)
        return 0;

    auto *macModule = interfaceModule->getSubmodule("mac");
    auto *ieee80211Mac = dynamic_cast<ieee80211::Ieee80211Mac *>(macModule);
    if (ieee80211Mac == nullptr)
        return 0;

    return ieee80211Mac->getBdStationCount();
}

void Aodv::logCbrRrepMetrics1s()
{
    if (!cbrRrepMetricsEnabled || pwd.empty())
        return;

    std::string externalId = "";
    auto parent = getParentModule();
    if (parent != nullptr && parent->hasPar("externalId"))
        externalId = parent->par("externalId").stdstringValue();

    cModule *nodeModule = getContainingNode(this);
    int nodeIndex = nodeModule != nullptr ? nodeModule->getIndex() : -1;
    double localCbr = getLocalCbr();
    int neighborCount = countCurrentNeighbors();
    auto activeRange = getActiveCbrThresholdRange();
    double blockRate = metricsRrepCandidateCount > 0 ? (double)metricsRrepBlockedCount / metricsRrepCandidateCount : 0.0;
    double discoveryDelayAvgMs = metricsRouteDiscoveryDelayCount > 0 ? 1000.0 * metricsRouteDiscoveryDelaySum.dbl() / metricsRouteDiscoveryDelayCount : 0.0;
    double routeCandidateAvg = metricsRouteCandidateCountCount > 0 ? (double)metricsRouteCandidateCountSum / metricsRouteCandidateCountCount : 0.0;
    double selectedHopAvg = metricsSelectedRouteHopCountCount > 0 ? (double)metricsSelectedRouteHopCountSum / metricsSelectedRouteHopCountCount : 0.0;

    std::filesystem::create_directories(pwd);
    std::string filePath = pwd + "/aodv_cbr_rrep_metrics_1s.csv";
    bool writeHeader = !std::filesystem::exists(filePath) || std::filesystem::file_size(filePath) == 0;
    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    if (writeHeader) {
        out << "time,node,nodeIndex,externalId,localCbr,neighborCount,appliedLowThreshold,appliedHighThreshold,"
               "rreqReceived,rrepCandidates,rrepAllowed,rrepBlocked,rrepBlockRate,"
               "routeDiscoveryStarted,routeDiscoverySucceeded,routeDiscoveryFailed,"
               "routeDiscoveryDelayAvgMs,rrepReceived,routeCandidateAvg,"
               "selectedRouteHopAvg,relayParticipation\n";
    }

    out << simTime() << ","
        << getParentModule()->getFullName() << ","
        << nodeIndex << ","
        << externalId << ","
        << localCbr << ","
        << neighborCount << ","
        << activeRange.first << ","
        << activeRange.second << ","
        << metricsRreqReceivedCount << ","
        << metricsRrepCandidateCount << ","
        << metricsRrepAllowedCount << ","
        << metricsRrepBlockedCount << ","
        << blockRate << ","
        << metricsRouteDiscoveryStartedCount << ","
        << metricsRouteDiscoverySucceededCount << ","
        << metricsRouteDiscoveryFailedCount << ","
        << discoveryDelayAvgMs << ","
        << metricsRrepReceivedCount << ","
        << routeCandidateAvg << ","
        << selectedHopAvg << ","
        << metricsRelayParticipationCount << "\n";

    metricsRreqReceivedCount = 0;
    metricsRrepCandidateCount = 0;
    metricsRrepAllowedCount = 0;
    metricsRrepBlockedCount = 0;
    metricsRouteDiscoveryStartedCount = 0;
    metricsRouteDiscoverySucceededCount = 0;
    metricsRouteDiscoveryFailedCount = 0;
    metricsRrepReceivedCount = 0;
    metricsRelayParticipationCount = 0;
    metricsRouteCandidateCountSum = 0;
    metricsRouteCandidateCountCount = 0;
    metricsSelectedRouteHopCountSum = 0;
    metricsSelectedRouteHopCountCount = 0;
    metricsRouteDiscoveryDelaySum = SIMTIME_ZERO;
    metricsRouteDiscoveryDelayCount = 0;
}

void Aodv::logTransmissionFailureDiagnosis1s()
{
    if (!transmissionFailureDiagnosisLogEnabled || pwd.empty())
        return;

    std::filesystem::create_directories(pwd);
    std::string filePath = pwd + "/aodv_transmission_failure_diagnosis_1s.csv";
    bool writeHeader = !std::filesystem::exists(filePath) || std::filesystem::file_size(filePath) == 0;
    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    double localCbr = getLocalCbr();
    int neighborCount = countCurrentNeighbors();
    auto activeRange = getActiveCbrThresholdRange();
    double routeDiscoveryDelayAvgMs = metricsRouteDiscoveryDelayCount > 0 ? 1000.0 * metricsRouteDiscoveryDelaySum.dbl() / metricsRouteDiscoveryDelayCount : 0.0;
    double rrepBlockRate = metricsRrepCandidateCount > 0 ? (double)metricsRrepBlockedCount / metricsRrepCandidateCount : 0.0;

    if (writeHeader) {
        out << "time,node,localCbr,neighborCount,appliedLowThreshold,appliedHighThreshold,"
               "routeDiscoveryStarted,routeDiscoverySucceeded,routeDiscoveryFailed,routeDiscoveryDelayAvgMs,"
               "rreqReceived,rrepReceived,rrepCandidates,rrepAllowed,rrepBlocked,rrepBlockRate,"
               "noRouteToForward,noActiveRouteToForward,"
               "routeInvalidate,routeExpireInactive,routeDelete,rerrOriginated\n";
    }

    out << simTime() << ","
        << getParentModule()->getFullName() << ","
        << localCbr << ","
        << neighborCount << ","
        << activeRange.first << ","
        << activeRange.second << ","
        << metricsRouteDiscoveryStartedCount << ","
        << metricsRouteDiscoverySucceededCount << ","
        << metricsRouteDiscoveryFailedCount << ","
        << routeDiscoveryDelayAvgMs << ","
        << metricsRreqReceivedCount << ","
        << metricsRrepReceivedCount << ","
        << metricsRrepCandidateCount << ","
        << metricsRrepAllowedCount << ","
        << metricsRrepBlockedCount << ","
        << rrepBlockRate << ","
        << diagnosisNoRouteToForwardCount << ","
        << diagnosisNoActiveRouteToForwardCount << ","
        << diagnosisRouteInvalidateCount << ","
        << diagnosisRouteExpireInactiveCount << ","
        << diagnosisRouteDeleteCount << ","
        << diagnosisRerrOriginatedCount << "\n";

    diagnosisNoRouteToForwardCount = 0;
    diagnosisNoActiveRouteToForwardCount = 0;
    diagnosisRouteInvalidateCount = 0;
    diagnosisRouteExpireInactiveCount = 0;
    diagnosisRouteDeleteCount = 0;
    diagnosisRerrOriginatedCount = 0;
}

void Aodv::ensureCbrRrepDecisionLogFile() const
{
    if (!cbrRrepDecisionLogEnabled || pwd.empty())
        return;

    std::filesystem::create_directories(pwd);
    std::string filePath = pwd + "/aodv_cbr_rrep_decisions.csv";
    bool writeHeader = !std::filesystem::exists(filePath) || std::filesystem::file_size(filePath) == 0;
    if (!writeHeader)
        return;

    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;
    out << "time,node,source,originator,destination,rreqId,hopCount,localCbr,threshold,appliedLowThreshold,appliedHighThreshold,decision\n";
}

void Aodv::logCbrRrepDecision(const Ptr<Rreq>& rreq, const L3Address& sourceAddr, double localCbr, const char *decision, double appliedLowThreshold, double appliedHighThreshold) const
{
    if (!cbrRrepDecisionLogEnabled || pwd.empty())
        return;

    ensureCbrRrepDecisionLogFile();
    std::string filePath = pwd + "/aodv_cbr_rrep_decisions.csv";
    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    out << simTime() << ","
        << getParentModule()->getFullName() << ","
        << sourceAddr << ","
        << rreq->getOriginatorAddr() << ","
        << rreq->getDestAddr() << ","
        << rreq->getRreqId() << ","
        << rreq->getHopCount() << ","
        << localCbr << ","
        << cbrBasedRrepThreshold << ","
        << appliedLowThreshold << ","
        << appliedHighThreshold << ","
        << decision << "\n";
}

void Aodv::ensureDlDirectThresholdRrepDebugLogFile() const
{
    if (!dlDirectThresholdRrepDebugLogEnabled || pwd.empty())
        return;

    std::filesystem::create_directories(pwd);
    std::string filePath = pwd + "/aodv_dl_direct_threshold_debug.csv";
    bool writeHeader = !std::filesystem::exists(filePath) || std::filesystem::file_size(filePath) == 0;
    if (!writeHeader)
        return;

    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    out << "time,node,source,originator,destination,rreqId,hopCount,localCbr,neighborCount,isDirectRoute,input0,input1,input2,input3,rawLow,rawHigh,predictedLow,predictedHigh,decision\n";
}

void Aodv::logDlDirectThresholdRrepDebug(const Ptr<Rreq>& rreq, const L3Address& sourceAddr, double localCbr, int neighborCount, unsigned int hopCount, bool isDirectRouteToDestination, const std::vector<double>& inputs, const std::array<double, 2>& rawOutputs, double predictedLow, double predictedHigh, const char *decision) const
{
    if (!dlDirectThresholdRrepDebugLogEnabled || pwd.empty())
        return;

    ensureDlDirectThresholdRrepDebugLogFile();
    std::string filePath = pwd + "/aodv_dl_direct_threshold_debug.csv";
    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    out << simTime() << ","
        << getParentModule()->getFullName() << ","
        << sourceAddr << ","
        << rreq->getOriginatorAddr() << ","
        << rreq->getDestAddr() << ","
        << rreq->getRreqId() << ","
        << hopCount << ","
        << localCbr << ","
        << neighborCount << ","
        << (isDirectRouteToDestination ? 1 : 0) << ","
        << (inputs.size() > 0 ? inputs[0] : 0.0) << ","
        << (inputs.size() > 1 ? inputs[1] : 0.0) << ","
        << (inputs.size() > 2 ? inputs[2] : 0.0) << ","
        << (inputs.size() > 3 ? inputs[3] : 0.0) << ","
        << rawOutputs[0] << ","
        << rawOutputs[1] << ","
        << predictedLow << ","
        << predictedHigh << ","
        << decision << "\n";
}

void Aodv::logRouteCauseEvent(const char *event, const L3Address& routeDest, const L3Address& nextHop, unsigned int hopCount, bool isActive, simtime_t lifeTime, const char *reason) const
{
    if (!cbrRouteCauseLogEnabled || pwd.empty())
        return;

    std::filesystem::create_directories(pwd);
    std::string filePath = pwd + "/aodv_route_cause_events.csv";
    bool writeHeader = !std::filesystem::exists(filePath) || std::filesystem::file_size(filePath) == 0;
    std::ofstream out(filePath, std::ios::app);
    if (!out.is_open())
        return;

    if (writeHeader)
        out << "time,node,event,routeDest,nextHop,hopCount,active,lifeTime,localCbr,reason\n";

    out << simTime() << ","
        << getParentModule()->getFullName() << ","
        << event << ","
        << routeDest << ","
        << nextHop << ","
        << hopCount << ","
        << (isActive ? 1 : 0) << ","
        << lifeTime << ","
        << getLocalCbr() << ","
        << reason << "\n";
}

void Aodv::handleRREQ(const Ptr<Rreq>& rreq, const L3Address& sourceAddr, unsigned int timeToLive)
{
    EV_INFO << "AODV Route Request arrived with source addr: " << sourceAddr << " originator addr: " << rreq->getOriginatorAddr()
                    << " destination addr: " << rreq->getDestAddr() << endl;
    if (cbrRrepMetricsEnabled)
        metricsRreqReceivedCount++;

    /*appendAodvMetric("aodv_control_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RREQ_RECV, source=" + sourceAddr.str() +
            ", originator=" + rreq->getOriginatorAddr().str() +
            ", target=" + rreq->getDestAddr().str() +
            ", ttl=" + std::to_string(timeToLive) +
            ", rreqId=" + std::to_string(rreq->getRreqId()));*/

    // A node ignores all RREQs received from any node in its blacklist set.

    if (containsKey(blacklist, sourceAddr)) {
        auto activeRange = getActiveCbrThresholdRange();
        logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "blacklist_drop", activeRange.first, activeRange.second);
        EV_INFO << "The sender node " << sourceAddr << " is in our blacklist. Ignoring the Route Request" << endl;
        return;
    }

    // When a node receives a RREQ, it first creates or updates a route to
    // the previous hop without a valid sequence number (see section 6.2).

    IRoute *previousHopRoute = routingTable->findBestMatchingRoute(sourceAddr);

    if (!previousHopRoute || previousHopRoute->getSource() != this) {
        // create without valid sequence number
        previousHopRoute = createRoute(sourceAddr, sourceAddr, 1, false, rreq->getOriginatorSeqNum(), true, simTime() + activeRouteTimeout);
    }
    else
        updateRoutingTable(previousHopRoute, sourceAddr, 1, false, rreq->getOriginatorSeqNum(), true, simTime() + activeRouteTimeout);

    // then checks to determine whether it has received a RREQ with the same
    // Originator IP Address and RREQ ID within at least the last PATH_DISCOVERY_TIME.
    // If such a RREQ has been received, the node silently discards the newly received RREQ.

    RreqIdentifier rreqIdentifier(rreq->getOriginatorAddr(), rreq->getRreqId());
    auto checkRREQArrivalTime = rreqsArrivalTime.find(rreqIdentifier);
    if (checkRREQArrivalTime != rreqsArrivalTime.end() && simTime() - checkRREQArrivalTime->second <= pathDiscoveryTime) {
        /*
        appendAodvMetric("aodv_control_log.txt",
                "time=" + simTime().str() +
                ", node=" + std::string(getParentModule()->getFullName()) +
                ", event=RREQ_DUPLICATE_DROP, source=" + sourceAddr.str() +
                ", originator=" + rreq->getOriginatorAddr().str() +
                ", target=" + rreq->getDestAddr().str() +
                ", rreqId=" + std::to_string(rreq->getRreqId()));*/
        auto activeRange = getActiveCbrThresholdRange();
        logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "duplicate_drop", activeRange.first, activeRange.second);
        EV_WARN << "The same packet has arrived within PATH_DISCOVERY_TIME= " << pathDiscoveryTime << ". Discarding it" << endl;
        return;
    }

    // update or create
    rreqsArrivalTime[rreqIdentifier] = simTime();
    summaryRreqAcceptCount++;
    if (enableRreqGraphLog) {
        appendAodvMetric("aodv_rreq_graph_log.csv",
                "time=" + simTime().str() +
                ",node=" + std::string(getParentModule()->getFullName()) +
                ",event=RREQ_ACCEPT" +
                ",source=" + sourceAddr.str() +
                ",originator=" + rreq->getOriginatorAddr().str() +
                ",target=" + rreq->getDestAddr().str() +
                ",rreqId=" + std::to_string(rreq->getRreqId()) +
                ",ttl=" + std::to_string(timeToLive));
    }

    // First, it first increments the hop count value in the RREQ by one, to
    // account for the new hop through the intermediate node.

    rreq->setHopCount(rreq->getHopCount() + 1);

    // Then the node searches for a reverse route to the Originator IP Address (see
    // section 6.2), using longest-prefix matching.

    IRoute *reverseRoute = routingTable->findBestMatchingRoute(rreq->getOriginatorAddr());

    // If need be, the route is created, or updated using the Originator Sequence Number from the
    // RREQ in its routing table.
    //
    // When the reverse route is created or updated, the following actions on
    // the route are also carried out:
    //
    //   1. the Originator Sequence Number from the RREQ is compared to the
    //      corresponding destination sequence number in the route table entry
    //      and copied if greater than the existing value there
    //
    //   2. the valid sequence number field is set to true;
    //
    //   3. the next hop in the routing table becomes the node from which the
    //      RREQ was received (it is obtained from the source IP address in
    //      the IP header and is often not equal to the Originator IP Address
    //      field in the RREQ message);
    //
    //   4. the hop count is copied from the Hop Count in the RREQ message;
    //
    //   Whenever a RREQ message is received, the Lifetime of the reverse
    //   route entry for the Originator IP address is set to be the maximum of
    //   (ExistingLifetime, MinimalLifetime), where
    //
    //   MinimalLifetime = (current time + 2*NET_TRAVERSAL_TIME - 2*HopCount*NODE_TRAVERSAL_TIME).

    unsigned int hopCount = rreq->getHopCount();
    simtime_t minimalLifeTime = simTime() + 2 * netTraversalTime - 2 * hopCount * nodeTraversalTime;
    simtime_t newLifeTime = std::max(simTime(), minimalLifeTime);
    int rreqSeqNum = rreq->getOriginatorSeqNum();
    if (!reverseRoute || reverseRoute->getSource() != this) { // create
        // This reverse route will be needed if the node receives a RREP back to the
        // node that originated the RREQ (identified by the Originator IP Address).
        reverseRoute = createRoute(rreq->getOriginatorAddr(), sourceAddr, hopCount, true, rreqSeqNum, true, newLifeTime);
    }
    else {
        AodvRouteData *routeData = check_and_cast<AodvRouteData *>(reverseRoute->getProtocolData());
        int routeSeqNum = routeData->getDestSeqNum();
        int newSeqNum = std::max(routeSeqNum, rreqSeqNum);
        int newHopCount = rreq->getHopCount(); // Note: already incremented by 1.
        int routeHopCount = reverseRoute->getMetric();
        // The route is only updated if the new sequence number is either
        //
        //   (i)       higher than the destination sequence number in the route
        //             table, or
        //
        //   (ii)      the sequence numbers are equal, but the hop count (of the
        //             new information) plus one, is smaller than the existing hop
        //             count in the routing table, or
        //
        //   (iii)     the sequence number is unknown.

        if (rreqSeqNum > routeSeqNum ||
                (rreqSeqNum == routeSeqNum && newHopCount < routeHopCount) ||
                rreq->getUnknownSeqNumFlag())
        {
            updateRoutingTable(reverseRoute, sourceAddr, hopCount, true, newSeqNum, true, newLifeTime);
        }
    }

    // A node generates a RREP if either:
    //
    // (i)       it is itself the destination, or
    //
    // (ii)      it has an active route to the destination, the destination
    //           sequence number in the node's existing route table entry
    //           for the destination is valid and greater than or equal to
    //           the Destination Sequence Number of the RREQ (comparison
    //           using signed 32-bit arithmetic), and the "destination only"
    //           ('D') flag is NOT set.

    // After a node receives a RREQ and responds with a RREP, it discards
    // the RREQ.  If the RREQ has the 'G' flag set, and the intermediate
    // node returns a RREP to the originating node, it MUST also unicast a
    // gratuitous RREP to the destination node.

    IRoute *destRoute = routingTable->findBestMatchingRoute(rreq->getDestAddr());
    AodvRouteData *destRouteData = destRoute ? dynamic_cast<AodvRouteData *>(destRoute->getProtocolData()) : nullptr;

    // check (i)
    if (rreq->getDestAddr() == getSelfIPAddress()) {
        auto activeRange = getActiveCbrThresholdRange();
        logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "destination_reply", activeRange.first, activeRange.second);
        EV_INFO << "I am the destination node for which the route was requested" << endl;

        // create RREP
        auto rrep = createRREP(rreq, destRoute, reverseRoute, sourceAddr);

        // send to the originator
        sendRREP(rrep, rreq->getOriginatorAddr(), 255);

        return; // discard RREQ, in this case, we do not forward it.
    }

    // check (ii)
    if (destRouteData && destRouteData->isActive() && destRouteData->hasValidDestNum() &&
            destRouteData->getDestSeqNum() >= rreq->getDestSeqNum())
    {
        EV_INFO << "I am an intermediate node who has information about a route to " << rreq->getDestAddr() << endl;

        if (destRoute->getNextHopAsGeneric() == sourceAddr) {
            auto activeRange = getActiveCbrThresholdRange();
            logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "loop_drop", activeRange.first, activeRange.second);
            EV_WARN << "This RREP would make a loop. Dropping it" << endl;
            return;
        }

        // we respond to the RREQ, if the D (destination only) flag is not set
        if (!rreq->getDestOnlyFlag()) {
            if (cbrRrepMetricsEnabled)
                metricsRrepCandidateCount++;
            double localCbr = getLocalCbr();
            int currentNeighborCount = countCurrentNeighbors();
            unsigned int currentRreqHopCount = rreq->getHopCount();
            bool isDirectRouteToDestination = destRoute->getMetric() <= 1 || destRoute->getNextHopAsGeneric() == rreq->getDestAddr();
            double activeLowThreshold = cbrBasedRrepLowThreshold;
            double activeHighThreshold = cbrBasedRrepHighThresholdForRange;
            bool blockedByCbrRange = isOutsideConfiguredCbrRange(localCbr, activeLowThreshold, activeHighThreshold);
            if (blockedByCbrRange && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                if (cbrRrepMetricsEnabled)
                    metricsRrepBlockedCount++;
                logCbrRrepDecision(rreq, sourceAddr, localCbr, "range_blocked", activeLowThreshold, activeHighThreshold);
                EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                        << " is outside allowed range (" << activeLowThreshold
                        << ", " << activeHighThreshold << ")" << endl;
                return;
            }
            if (dlBucketBasedRrepEnabled) {
                auto predictedRange = inferDlBucketBasedRrepThresholdRange(localCbr, currentNeighborCount, currentRreqHopCount);
                bool blockedByDlBucketMode = !(predictedRange.first < localCbr && localCbr < predictedRange.second);
                if (blockedByDlBucketMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "dl_bucket_blocked", predictedRange.first, predictedRange.second);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is outside bucket-predicted range (" << predictedRange.first
                            << ", " << predictedRange.second << ")" << endl;
                    return;
                }
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByDlBucketMode) ? "dl_bucket_direct_bypass_allow" : "dl_bucket_allowed", predictedRange.first, predictedRange.second);
            }
            else if (stateLookupBasedRrepEnabled) {
                auto predictedRange = inferStateLookupBasedRrepThresholdRange(localCbr, currentNeighborCount, currentRreqHopCount, isDirectRouteToDestination);
                bool blockedByStateLookupMode = !(predictedRange.first < localCbr && localCbr < predictedRange.second);
                if (blockedByStateLookupMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "state_lookup_blocked", predictedRange.first, predictedRange.second);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is outside state-lookup-predicted range (" << predictedRange.first
                            << ", " << predictedRange.second << ")" << endl;
                    return;
                }
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByStateLookupMode) ? "state_lookup_direct_bypass_allow" : "state_lookup_allowed", predictedRange.first, predictedRange.second);
            }
            else if (treeBasedRrepEnabled) {
                auto predictedRange = inferTreeBasedRrepThresholdRange(localCbr, currentNeighborCount, currentRreqHopCount, isDirectRouteToDestination);
                bool blockedByTreeMode = !(predictedRange.first < localCbr && localCbr < predictedRange.second);
                if (blockedByTreeMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "tree_blocked", predictedRange.first, predictedRange.second);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is outside tree-predicted range (" << predictedRange.first
                            << ", " << predictedRange.second << ")" << endl;
                    return;
                }
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByTreeMode) ? "tree_direct_bypass_allow" : "tree_allowed", predictedRange.first, predictedRange.second);
            }
            else if (dlDirectThresholdRrepEnabled) {
                std::vector<double> debugInputs;
                std::array<double, 2> debugRawOutputs = {};
                auto predictedRange = inferDlDirectThresholdRrepThresholdRange(localCbr, currentNeighborCount, destRoute->getMetric(), isDirectRouteToDestination, &debugInputs, &debugRawOutputs);
                bool blockedByDlDirectThresholdMode = !(predictedRange.first < localCbr && localCbr < predictedRange.second);
                if (blockedByDlDirectThresholdMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logDlDirectThresholdRrepDebug(rreq, sourceAddr, localCbr, currentNeighborCount, destRoute->getMetric(), isDirectRouteToDestination, debugInputs, debugRawOutputs, predictedRange.first, predictedRange.second, "dl_direct_threshold_blocked");
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "dl_direct_threshold_blocked", predictedRange.first, predictedRange.second);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is outside direct-threshold-DL-predicted range (" << predictedRange.first
                            << ", " << predictedRange.second << ")" << endl;
                    return;
                }
                logDlDirectThresholdRrepDebug(rreq, sourceAddr, localCbr, currentNeighborCount, destRoute->getMetric(), isDirectRouteToDestination, debugInputs, debugRawOutputs, predictedRange.first, predictedRange.second, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByDlDirectThresholdMode) ? "dl_direct_threshold_direct_bypass_allow" : "dl_direct_threshold_allowed");
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByDlDirectThresholdMode) ? "dl_direct_threshold_direct_bypass_allow" : "dl_direct_threshold_allowed", predictedRange.first, predictedRange.second);
            }
            else if (dlBasedRrepEnabled) {
                auto predictedRange = inferDlBasedRrepThresholdRange(localCbr, currentNeighborCount, destRoute->getMetric(), isDirectRouteToDestination);
                bool blockedByDlMode = !(predictedRange.first < localCbr && localCbr < predictedRange.second);
                if (blockedByDlMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "dl_blocked", predictedRange.first, predictedRange.second);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is outside DL-predicted range (" << predictedRange.first
                            << ", " << predictedRange.second << ")" << endl;
                    return;
                }
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByDlMode) ? "dl_direct_bypass_allow" : "dl_allowed", predictedRange.first, predictedRange.second);
            }
            else if (cbrBasedRrepEnabled) {
                bool blockedByCbrMode = shouldBlockByMode(localCbr, cbrBasedRrepThreshold, cbrBasedRrepCompareMode);
                if (blockedByCbrMode && !(cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination)) {
                    if (cbrRrepMetricsEnabled)
                        metricsRrepBlockedCount++;
                    logCbrRrepDecision(rreq, sourceAddr, localCbr, "blocked", cbrBasedRrepThreshold, cbrBasedRrepThreshold);
                    EV_INFO << "Skipping intermediate RREP because local CBR " << localCbr
                            << " is " << describeModeRelation(cbrBasedRrepCompareMode)
                            << " threshold " << cbrBasedRrepThreshold << endl;
                    return;
                }
                logCbrRrepDecision(rreq, sourceAddr, localCbr, cbrBasedRrepDirectRouteBypassEnabled && isDirectRouteToDestination && (blockedByCbrRange || blockedByCbrMode) ? "direct_bypass_allow" : "allowed", cbrBasedRrepThreshold, cbrBasedRrepThreshold);
            }
            else {
                logCbrRrepDecision(rreq, sourceAddr, localCbr, "disabled_allow", activeLowThreshold, activeHighThreshold);
            }
            if (cbrRrepMetricsEnabled) {
                metricsRrepAllowedCount++;
                metricsRelayParticipationCount++;
            }
            if (destRoute == nullptr || reverseRoute == nullptr || destRoute->getSource() != this || reverseRoute->getSource() != this) {
                if (cbrRrepMetricsEnabled)
                    metricsRrepBlockedCount++;
                logCbrRrepDecision(rreq, sourceAddr, localCbr, "route_missing_before_rrep", activeLowThreshold, activeHighThreshold);
                EV_WARN << "Skipping intermediate RREP because destination/reverse route is unavailable just before RREP creation. "
                        << "destRoute=" << (destRoute != nullptr)
                        << ", reverseRoute=" << (reverseRoute != nullptr) << endl;
                return;
            }
            simtime_t intermediateRrepDelay = computeIntermediateRrepDelay(localCbr, isDirectRouteToDestination);
            // create RREP
            auto rrep = createRREP(rreq, destRoute, reverseRoute, sourceAddr);

            // send to the originator
            sendRREP(rrep, rreq->getOriginatorAddr(), 255, intermediateRrepDelay);

            if (rreq->getGratuitousRREPFlag()) {
                // The gratuitous RREP is then sent to the next hop along the path to
                // the destination node, just as if the destination node had already
                // issued a RREQ for the originating node and this RREP was produced in
                // response to that (fictitious) RREQ.

                IRoute *originatorRoute = routingTable->findBestMatchingRoute(rreq->getOriginatorAddr());
                if (originatorRoute == nullptr || originatorRoute->getSource() != this) {
                    EV_WARN << "Skipping gratuitous RREP because originator route no longer exists for "
                            << rreq->getOriginatorAddr() << endl;
                }
                else {
                    auto grrep = createGratuitousRREP(rreq, originatorRoute);
                    sendGRREP(grrep, rreq->getDestAddr(), 100);
                }
            }

            return; // discard RREQ, in this case, we also do not forward it.
        }
        else {
            auto activeRange = getActiveCbrThresholdRange();
            logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "dest_only_forward", activeRange.first, activeRange.second);
            EV_INFO << "The originator indicated that only the destination may respond to this RREQ (D flag is set). Forwarding ..." << endl;
        }
    }

    // If a node does not generate a RREP (following the processing rules in
    // section 6.6), and if the incoming IP header has TTL larger than 1,
    // the node updates and broadcasts the RREQ to address 255.255.255.255
    // on each of its configured interfaces (see section 6.14).  To update
    // the RREQ, the TTL or hop limit field in the outgoing IP header is
    // decreased by one, and the Hop Count field in the RREQ message is
    // incremented by one, to account for the new hop through the
    // intermediate node. (!) Lastly, the Destination Sequence number for the
    // requested destination is set to the maximum of the corresponding
    // value received in the RREQ message, and the destination sequence
    // value currently maintained by the node for the requested destination.
    // However, the forwarding node MUST NOT modify its maintained value for
    // the destination sequence number, even if the value received in the
    // incoming RREQ is larger than the value currently maintained by the
    // forwarding node.

    if (timeToLive > 0 && (simTime() > rebootTime + deletePeriod || rebootTime == 0)) {
        if (destRouteData)
            rreq->setDestSeqNum(std::max(destRouteData->getDestSeqNum(), rreq->getDestSeqNum()));
        rreq->setUnknownSeqNumFlag(false);

        auto outgoingRREQ = dynamicPtrCast<Rreq>(rreq->dupShared());
        if (cbrRrepMetricsEnabled)
            metricsRelayParticipationCount++;
        auto activeRange = getActiveCbrThresholdRange();
        logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "forward_only", activeRange.first, activeRange.second);
        forwardRREQ(outgoingRREQ, timeToLive);
    }
    else {
        auto activeRange = getActiveCbrThresholdRange();
        logCbrRrepDecision(rreq, sourceAddr, getLocalCbr(), "ttl_drop", activeRange.first, activeRange.second);
        EV_WARN << "Can't forward the RREQ because of its small (<= 1) TTL: " << timeToLive << " or the AODV reboot has not completed yet" << endl;
    }
}

IRoute *Aodv::createRoute(const L3Address& destAddr, const L3Address& nextHop,
        unsigned int hopCount, bool hasValidDestNum, unsigned int destSeqNum,
        bool isActive, simtime_t lifeTime)
{
    // create a new route
    IRoute *newRoute = routingTable->createRoute();

    // adding generic fields
    newRoute->setDestination(destAddr);
    newRoute->setNextHop(nextHop);
    newRoute->setPrefixLength(addressType->getMaxPrefixLength()); // TODO
    newRoute->setMetric(hopCount);
    NetworkInterface *ifEntry = interfaceTable->findInterfaceByName(par("interface")); // TODO IMPLEMENT: multiple interfaces
    if (ifEntry)
        newRoute->setInterface(ifEntry);
    newRoute->setSourceType(IRoute::AODV);
    newRoute->setSource(this);

    // A route towards a destination that has a routing table entry
    // that is marked as valid.  Only active routes can be used to
    // forward data packets.

    // adding protocol-specific fields
    AodvRouteData *newProtocolData = new AodvRouteData();
    newProtocolData->setIsActive(isActive);
    newProtocolData->setHasValidDestNum(hasValidDestNum);
    newProtocolData->setDestSeqNum(destSeqNum);
    newProtocolData->setLifeTime(lifeTime);
    newRoute->setProtocolData(newProtocolData);

    EV_DETAIL << "Adding new route " << newRoute << endl;
    routingTable->addRoute(newRoute);
    logRouteGraphEvent("ROUTE_CREATE", destAddr, nextHop, hopCount, isActive, lifeTime);
    logRouteCauseEvent("ROUTE_CREATE", destAddr, nextHop, hopCount, isActive, lifeTime, "createRoute");
    logRoutingTableSnapshot("ROUTE_CREATE");

    scheduleExpungeRoutes();
    return newRoute;
}

void Aodv::receiveSignal(cComponent *source, simsignal_t signalID, cObject *obj, cObject *details)
{
    Enter_Method("%s", cComponent::getSignalName(signalID));

    if (signalID == linkBrokenSignal) {
        EV_DETAIL << "Received link break signal" << endl;
        Packet *datagram = check_and_cast<Packet *>(obj);
        const auto& networkHeader = findNetworkProtocolHeader(datagram);
        if (networkHeader != nullptr) {
            L3Address unreachableAddr = networkHeader->getDestinationAddress();
            if (unreachableAddr.getAddressType() == addressType) {
                // A node initiates processing for a RERR message in three situations:
                //
                //   (i)     if it detects a link break for the next hop of an active
                //           route in its routing table while transmitting data (and
                //           route repair, if attempted, was unsuccessful), or

                // TODO Implement: local repair

                IRoute *route = routingTable->findBestMatchingRoute(unreachableAddr);

                if (route && route->getSource() == this)
                    handleLinkBreakSendRERR(route->getNextHopAsGeneric());
            }
        }
    }
}

void Aodv::handleLinkBreakSendRERR(const L3Address& unreachableAddr)
{
    // For case (i), the node first makes a list of unreachable destinations
    // consisting of the unreachable neighbor and any additional
    // destinations (or subnets, see section 7) in the local routing table
    // that use the unreachable neighbor as the next hop.

    // Just before transmitting the RERR, certain updates are made on the
    // routing table that may affect the destination sequence numbers for
    // the unreachable destinations.  For each one of these destinations,
    // the corresponding routing table entry is updated as follows:
    //
    // 1. The destination sequence number of this routing entry, if it
    //    exists and is valid, is incremented for cases (i) and (ii) above,
    //    and copied from the incoming RERR in case (iii) above.
    //
    // 2. The entry is invalidated by marking the route entry as invalid
    //
    // 3. The Lifetime field is updated to current time plus DELETE_PERIOD.
    //    Before this time, the entry SHOULD NOT be deleted.

    IRoute *unreachableRoute = routingTable->findBestMatchingRoute(unreachableAddr);

    if (!unreachableRoute || unreachableRoute->getSource() != this)
        return;

    std::vector<UnreachableNode> unreachableNodes;
    AodvRouteData *unreachableRouteData = dynamic_cast<AodvRouteData *>(unreachableRoute->getProtocolData());
    if (unreachableRouteData == nullptr) {
        EV_WARN << "Cannot send RERR because unreachable route protocol data is missing for "
                << unreachableAddr << endl;
        return;
    }

    if (unreachableRouteData->isActive()) {
        UnreachableNode node;
        node.addr = unreachableAddr;
        node.seqNum = unreachableRouteData->getDestSeqNum();
        unreachableNodes.push_back(node);
    }

    // For case (i), the node first makes a list of unreachable destinations
    // consisting of the unreachable neighbor and any additional destinations
    // (or subnets, see section 7) in the local routing table that use the
    // unreachable neighbor as the next hop. Test

    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);

        AodvRouteData *routeData = dynamic_cast<AodvRouteData *>(route->getProtocolData());
        if (routeData && routeData->isActive() && route->getNextHopAsGeneric() == unreachableAddr) {
            if (routeData->hasValidDestNum())
                routeData->setDestSeqNum(routeData->getDestSeqNum() + 1);

            EV_DETAIL << "Marking route to " << route->getDestinationAsGeneric() << " as inactive" << endl;

            routeData->setIsActive(false);
            routeData->setLifeTime(simTime() + deletePeriod);
            diagnosisRouteInvalidateCount++;
            logRouteCauseEvent("ROUTE_INVALIDATE", route->getDestinationAsGeneric(), route->getNextHopAsGeneric(), route->getMetric(), false, routeData->getLifeTime(), "local_link_break");
            scheduleExpungeRoutes();

            UnreachableNode node;
            node.addr = route->getDestinationAsGeneric();
            node.seqNum = routeData->getDestSeqNum();
            unreachableNodes.push_back(node);
        }
    }

    if (!unreachableNodes.empty())
        logRoutingTableSnapshot("LOCAL_LINK_BREAK_INVALIDATE");

    // The neighboring node(s) that should receive the RERR are all those
    // that belong to a precursor list of at least one of the unreachable
    // destination(s) in the newly created RERR.  In case there is only one
    // unique neighbor that needs to receive the RERR, the RERR SHOULD be
    // unicast toward that neighbor.  Otherwise the RERR is typically sent
    // to the local broadcast address (Destination IP == 255.255.255.255,
    // TTL == 1) with the unreachable destinations, and their corresponding
    // destination sequence numbers, included in the packet.

    if (rerrCount >= rerrRatelimit) {
        EV_WARN << "A node should not generate more than RERR_RATELIMIT RERR messages per second. Canceling sending RERR" << endl;
        return;
    }

    if (unreachableNodes.empty())
        return;

    std::set<L3Address> precursorNodes;
    for (const auto& unreachableNode : unreachableNodes) {
        IRoute *route = routingTable->findBestMatchingRoute(unreachableNode.addr);
        AodvRouteData *routeData = route ? dynamic_cast<AodvRouteData *>(route->getProtocolData()) : nullptr;
        if (routeData != nullptr) {
            for (const auto& precursorNode : routeData->getPrecursorList())
                precursorNodes.insert(precursorNode);
        }
    }

    logOriginatedRerr("LOCAL_LINK_BREAK", unreachableNodes, precursorNodes);
    diagnosisRerrOriginatedCount++;
    logRouteCauseEvent("RERR_ORIGINATED", unreachableAddr, L3Address(), 0, false, SIMTIME_ZERO, "local_link_break");
    auto rerr = createRERR(unreachableNodes);
    rerrCount++;

    // broadcast
    EV_INFO << "Broadcasting Route Error message with TTL=1" << endl;
    sendAODVPacket(rerr, addressType->getBroadcastAddress(), 1, *jitterPar);
}

const Ptr<Rerr> Aodv::createRERR(const std::vector<UnreachableNode>& unreachableNodes)
{
    auto rerr = makeShared<Rerr>(); // TODO "AODV-RERR");
    rerr->setPacketType(usingIpv6 ? RERR_IPv6 : RERR);

    unsigned int destCount = unreachableNodes.size();
    rerr->setUnreachableNodesArraySize(destCount);

    for (unsigned int i = 0; i < destCount; i++) {
        UnreachableNode node;
        node.addr = unreachableNodes[i].addr;
        node.seqNum = unreachableNodes[i].seqNum;
        rerr->setUnreachableNodes(i, node);
    }

    rerr->setChunkLength(B(4 + destCount * (usingIpv6 ? (4 + 16) : (4 + 4))));

    return rerr;
}

void Aodv::appendAodvMetric(const std::string& fileName, const std::string& line) const
{
    if (pwd.empty())
        return;
    std::ofstream logFile(pwd + "/" + fileName, std::ios::app);
    logFile << line << endl;
}

void Aodv::logPrecursorAddition(const char *reason, const L3Address& routeDest, const L3Address& precursor, const std::set<L3Address>& precursorList) const // @suppress("Member declaration not found")
{
    if (!enablePrecursorLog)
        return;
    appendAodvMetric("aodv_precursor_log.csv",
            "time=" + simTime().str() +
            ",node=" + std::string(getParentModule()->getFullName()) +
            ",event=PRECURSOR_ADD" +
            ",reason=" + reason +
            ",routeDest=" + routeDest.str() +
            ",addedPrecursor=" + precursor.str() +
            ",precursorCount=" + std::to_string(precursorList.size()) +
            ",precursors=" + joinAddresses(precursorList));
}

void Aodv::logRouteGraphEvent(const char *event, const L3Address& routeDest, const L3Address& nextHop, unsigned int hopCount, bool isActive, simtime_t lifeTime) const
{
    if (!enableRouteGraphLog)
        return;
    appendAodvMetric("aodv_route_graph_log.csv",
            "time=" + simTime().str() +
            ",node=" + std::string(getParentModule()->getFullName()) +
            ",event=" + event +
            ",routeDest=" + routeDest.str() +
            ",nextHop=" + nextHop.str() +
            ",hopCount=" + std::to_string(hopCount) +
            ",active=" + std::to_string(isActive ? 1 : 0) +
            ",lifeTime=" + lifeTime.str() +
            ",routeTableSize=" + std::to_string(routingTable->getNumRoutes()));
}

void Aodv::logRoutingTableSnapshot(const char *reason)
{
    if (!enableRoutingTableSnapshotLog)
        return;
    if (pwd.empty())
        return;
    if (simTime() < nextRoutingTableSnapshotTime)
        return;

    nextRoutingTableSnapshotTime = simTime() + 1;

    int managedRouteCount = 0;
    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        if (route->getSource() == this)
            managedRouteCount++;
    }

    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        if (route->getSource() != this)
            continue;

        auto routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
        const auto& precursorList = routeData->getPrecursorList();
        appendAodvMetric("aodv_routing_table_snapshot.csv",
                "time=" + simTime().str() +
                ",node=" + std::string(getParentModule()->getFullName()) +
                ",reason=" + reason +
                ",routeTableSize=" + std::to_string(managedRouteCount) +
                ",routeDest=" + route->getDestinationAsGeneric().str() +
                ",nextHop=" + route->getNextHopAsGeneric().str() +
                ",hopCount=" + std::to_string(route->getMetric()) +
                ",active=" + std::to_string(routeData->isActive() ? 1 : 0) +
                ",hasValidDestNum=" + std::to_string(routeData->hasValidDestNum() ? 1 : 0) +
                ",destSeqNum=" + std::to_string(routeData->getDestSeqNum()) +
                ",lifeTime=" + routeData->getLifeTime().str() +
                ",precursorCount=" + std::to_string(precursorList.size()) +
                ",precursors=" + joinAddresses(precursorList));
    }
}

void Aodv::logSummary1s()
{
    if (!enableSummary1sLog || pwd.empty())
        return;

    std::string externalId = "";
    auto host = getParentModule();
    if (host != nullptr && host->hasPar("externalId"))
        externalId = host->par("externalId").stdstringValue();
    int managedRouteCount = 0;
    int activeRouteCount = 0;
    int precursorSum = 0;

    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        if (route->getSource() != this)
            continue;

        managedRouteCount++;
        auto routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
        if (routeData->isActive())
            activeRouteCount++;
        precursorSum += routeData->getPrecursorList().size();
    }

    appendAodvMetric("aodv_summary_1s.csv",
            "time=" + simTime().str() +
            ",node=" + std::string(getParentModule()->getFullName()) +
            ",externalId=" + externalId +
            ",rreqAcceptCount=" + std::to_string(summaryRreqAcceptCount) +
            ",managedRouteCount=" + std::to_string(managedRouteCount) +
            ",activeRouteCount=" + std::to_string(activeRouteCount) +
            ",precursorSum=" + std::to_string(precursorSum) +
            ",rerrGeneratedCount=" + std::to_string(summaryRerrGeneratedCount) +
            ",rerrUnreachableSum=" + std::to_string(summaryRerrUnreachableSum) +
            ",rerrPrecursorSum=" + std::to_string(summaryRerrPrecursorSum));

    summaryRreqAcceptCount = 0;
    summaryRerrGeneratedCount = 0;
    summaryRerrUnreachableSum = 0;
    summaryRerrPrecursorSum = 0;
}

void Aodv::logOriginatedRerr(const char *reason, const std::vector<UnreachableNode>& unreachableNodes, const std::set<L3Address>& precursorNodes) // @suppress("Member declaration not found")
{
    summaryRerrGeneratedCount++;
    summaryRerrUnreachableSum += unreachableNodes.size();
    summaryRerrPrecursorSum += precursorNodes.size();

    if (!enableRerrFanoutLog)
        return;
    if (pwd.empty())
        return;

    appendAodvMetric("aodv_rerr_fanout_log.csv",
            "time=" + simTime().str() +
            ",node=" + std::string(getParentModule()->getFullName()) +
            ",event=RERR_FANOUT" +
            ",reason=" + reason +
            ",unreachableCount=" + std::to_string(unreachableNodes.size()) +
            ",precursorCount=" + std::to_string(precursorNodes.size()) +
            ",unreachable=" + joinUnreachableNodes(unreachableNodes) +
            ",precursors=" + joinAddresses(precursorNodes));

    // Match the existing multi-line debug format so RERR contents and precursor
    // lists can be inspected side by side with earlier experiment logs.

    totalOriginatedRerrCount++;

    std::ofstream logFileR(pwd + "/rerr_debug.txt", std::ios::app);
    logFileR << "[RERR 생성] 시간: " << simTime()
             << ", 노드: " << getParentModule()->getFullName()
             << ", 목적지 수: " << unreachableNodes.size() << std::endl;
    for (const auto& unreachableNode : unreachableNodes)
        logFileR << "  - 대상: " << unreachableNode.addr << ", SeqNum: " << unreachableNode.seqNum << std::endl;

    std::ofstream logFileP(pwd + "/rerr_precursor_log.txt", std::ios::app);
    logFileP << "[RERR 생성] 시간: " << simTime()
             << ", 노드: " << getParentModule()->getFullName()
             << ", Precursor 수: " << precursorNodes.size() << std::endl;
    for (const auto& precursorNode : precursorNodes)
        logFileP << "  - Precursor: " << precursorNode << std::endl;
}

void Aodv::handleRERR(const Ptr<const Rerr>& rerr, const L3Address& sourceAddr)
{
    EV_INFO << "AODV Route Error arrived with source addr: " << sourceAddr << endl;
    /*
    appendAodvMetric("aodv_control_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RERR_RECV, source=" + sourceAddr.str() +
            ", unreachableCount=" + std::to_string(rerr->getUnreachableNodesArraySize()));*/

    // A node initiates processing for a RERR message in three situations:
    // (iii)   if it receives a RERR from a neighbor for one or more
    //         active routes.
    unsigned int unreachableArraySize = rerr->getUnreachableNodesArraySize();
    std::vector<UnreachableNode> unreachableNeighbors;
    std::set<L3Address> precursorNodes;

    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        AodvRouteData *routeData = route ? dynamic_cast<AodvRouteData *>(route->getProtocolData()) : nullptr;

        if (!routeData)
            continue;

        // For case (iii), the list should consist of those destinations in the RERR
        // for which there exists a corresponding entry in the local routing
        // table that has the transmitter of the received RERR as the next hop.

        if (routeData->isActive() && route->getNextHopAsGeneric() == sourceAddr) {
            for (unsigned int j = 0; j < unreachableArraySize; j++) {
                if (route->getDestinationAsGeneric() == rerr->getUnreachableNodes(j).addr) {
                    // 1. The destination sequence number of this routing entry, if it
                    // exists and is valid, is incremented for cases (i) and (ii) above,
                    // ! and copied from the incoming RERR in case (iii) above.

                    routeData->setDestSeqNum(rerr->getUnreachableNodes(j).seqNum);
                    routeData->setIsActive(false); // it means invalid, see 3. AODV Terminology p.3. in RFC 3561
                    routeData->setLifeTime(simTime() + deletePeriod);
                    diagnosisRouteInvalidateCount++;
                    logRouteCauseEvent("ROUTE_INVALIDATE", route->getDestinationAsGeneric(), route->getNextHopAsGeneric(), route->getMetric(), false, routeData->getLifeTime(), "received_rerr");

                    // The RERR should contain those destinations that are part of
                    // the created list of unreachable destinations and have a non-empty
                    // precursor list.

                    if (routeData->getPrecursorList().size() > 0) {
                        UnreachableNode node;
                        node.addr = route->getDestinationAsGeneric();
                        node.seqNum = routeData->getDestSeqNum();
                        unreachableNeighbors.push_back(node);
                        for (const auto& precursorNode : routeData->getPrecursorList())
                            precursorNodes.insert(precursorNode);
                    }
                    scheduleExpungeRoutes();
                }
            }
        }
    }

    if (!unreachableNeighbors.empty())
        logRoutingTableSnapshot("HANDLE_RERR_INVALIDATE");

    if (rerrCount >= rerrRatelimit) {
        EV_WARN << "A node should not generate more than RERR_RATELIMIT RERR messages per second. Canceling sending RERR" << endl;
        return;
    }

    if (unreachableNeighbors.size() > 0 && (simTime() > rebootTime + deletePeriod || rebootTime == 0)) {
        EV_INFO << "Sending RERR to inform our neighbors about link breaks." << endl;
        logOriginatedRerr("FORWARDED_RERR", unreachableNeighbors, precursorNodes);
        diagnosisRerrOriginatedCount++;
        logRouteCauseEvent("RERR_ORIGINATED", sourceAddr, L3Address(), 0, false, SIMTIME_ZERO, "forwarded_rerr");
        auto newRERR = createRERR(unreachableNeighbors);
        sendAODVPacket(newRERR, addressType->getBroadcastAddress(), 1, 0);
        rerrCount++;
    }
}

void Aodv::handleStartOperation(LifecycleOperation *operation)
{
    rebootTime = simTime();

    socket.setOutputGate(gate("socketOut"));
    socket.setCallback(this);
    socket.bind(L3Address(), aodvUDPPort);
    socket.setBroadcast(true);

    // RFC 5148:
    // Jitter SHOULD be applied by reducing this delay by a random amount, so that
    // the delay between consecutive transmissions of messages of the same type is
    // equal to (MESSAGE_INTERVAL - jitter), where jitter is the random value.
    if (useHelloMessages)
        scheduleAfter(helloInterval - *periodicJitter, helloMsgTimer);

    scheduleAfter(1, counterTimer);
}

void Aodv::handleStopOperation(LifecycleOperation *operation)
{
    socket.close();
    clearState();
}

void Aodv::handleCrashOperation(LifecycleOperation *operation)
{
    socket.destroy();
    clearState();
}

void Aodv::clearState()
{
    rerrCount = rreqCount = rreqId = sequenceNum = 0;
    totalOriginatedRerrCount = 0;
    addressToRreqRetries.clear();
    metricsRreqReceivedCount = 0;
    metricsRrepCandidateCount = 0;
    metricsRrepAllowedCount = 0;
    metricsRrepBlockedCount = 0;
    metricsRouteDiscoveryStartedCount = 0;
    metricsRouteDiscoverySucceededCount = 0;
    metricsRouteDiscoveryFailedCount = 0;
    metricsRrepReceivedCount = 0;
    metricsRelayParticipationCount = 0;
    metricsRouteCandidateCountSum = 0;
    metricsRouteCandidateCountCount = 0;
    metricsSelectedRouteHopCountSum = 0;
    metricsSelectedRouteHopCountCount = 0;
    metricsRouteDiscoveryDelaySum = SIMTIME_ZERO;
    metricsRouteDiscoveryDelayCount = 0;
    metricsRouteDiscoveryStartTimes.clear();
    metricsRouteDiscoveryCandidateCounts.clear();
    cbrBasedRandomThresholdEpoch = -1;
    cbrBasedRandomActiveLowThreshold = 0;
    cbrBasedRandomActiveHighThreshold = 0;
    diagnosisNoRouteToForwardCount = 0;
    diagnosisNoActiveRouteToForwardCount = 0;
    diagnosisRouteInvalidateCount = 0;
    diagnosisRouteExpireInactiveCount = 0;
    diagnosisRouteDeleteCount = 0;
    diagnosisRerrOriginatedCount = 0;
    for (auto& elem : waitForRREPTimers)
        cancelAndDelete(elem.second);

    // FIXME Drop the queued datagrams.
    //    for (auto it = targetAddressToDelayedPackets.begin(); it != targetAddressToDelayedPackets.end(); it++)
    //       networkProtocol->dropQueuedDatagram(it->second);

    targetAddressToDelayedPackets.clear();

    waitForRREPTimers.clear();
    rreqsArrivalTime.clear();

    if (useHelloMessages)
        cancelEvent(helloMsgTimer);
    if (expungeTimer)
        cancelEvent(expungeTimer);
    if (counterTimer)
        cancelEvent(counterTimer);
    if (blacklistTimer)
        cancelEvent(blacklistTimer);
    if (rrepAckTimer)
        cancelEvent(rrepAckTimer);
}

void Aodv::handleWaitForRREP(WaitForRrep *rrepTimer)
{
    EV_INFO << "We didn't get any Route Reply within RREP timeout" << endl;
    L3Address destAddr = rrepTimer->getDestAddr();
    /*
    appendAodvMetric("aodv_route_log.txt",
            "time=" + simTime().str() +
            ", node=" + std::string(getParentModule()->getFullName()) +
            ", event=RREP_TIMEOUT, target=" + destAddr.str() +
            ", lastTTL=" + std::to_string(rrepTimer->getLastTTL()) +
            ", retryCount=" + std::to_string(addressToRreqRetries[destAddr]));*/

    ASSERT(containsKey(addressToRreqRetries, destAddr));
    if (addressToRreqRetries[destAddr] == rreqRetries) {
        if (cbrRrepMetricsEnabled) {
            metricsRouteDiscoveryFailedCount++;
            auto candidateIt = metricsRouteDiscoveryCandidateCounts.find(destAddr);
            if (candidateIt != metricsRouteDiscoveryCandidateCounts.end()) {
                metricsRouteCandidateCountSum += candidateIt->second;
                metricsRouteCandidateCountCount++;
            }
        }
        cancelRouteDiscovery(destAddr);
        EV_WARN << "Re-discovery attempts for node " << destAddr << " reached RREQ_RETRIES= " << rreqRetries << " limit. Stop sending RREQ." << endl;
        return;
    }

    auto rreq = createRREQ(destAddr);

    // the node MAY try again to discover a route by broadcasting another
    // RREQ, up to a maximum of RREQ_RETRIES times at the maximum TTL value.
    if (rrepTimer->getLastTTL() == netDiameter) // netDiameter is the maximum TTL value
        addressToRreqRetries[destAddr]++;

    sendRREQ(rreq, addressType->getBroadcastAddress(), 0);
}

void Aodv::forwardRREP(const Ptr<Rrep>& rrep, const L3Address& destAddr, unsigned int timeToLive)
{
    EV_INFO << "Forwarding the Route Reply to the node " << rrep->getOriginatorAddr() << " which originated the Route Request" << endl;

    // RFC 5148:
    // When a node forwards a message, it SHOULD be jittered by delaying it
    // by a random duration.  This delay SHOULD be generated uniformly in an
    // interval between zero and MAXJITTER.
    sendAODVPacket(rrep, destAddr, 100, *jitterPar);
}

void Aodv::forwardRREQ(const Ptr<Rreq>& rreq, unsigned int timeToLive)
{
    EV_INFO << "Forwarding the Route Request message with TTL= " << timeToLive << endl;
    sendAODVPacket(rreq, addressType->getBroadcastAddress(), timeToLive, *jitterPar);
}

void Aodv::completeRouteDiscovery(const L3Address& target)
{
    EV_DETAIL << "Completing route discovery, originator " << getSelfIPAddress() << ", target " << target << endl;
    ASSERT(hasOngoingRouteDiscovery(target));
    if (cbrRrepMetricsEnabled) {
        metricsRouteDiscoverySucceededCount++;
        auto startIt = metricsRouteDiscoveryStartTimes.find(target);
        if (startIt != metricsRouteDiscoveryStartTimes.end()) {
            metricsRouteDiscoveryDelaySum += simTime() - startIt->second;
            metricsRouteDiscoveryDelayCount++;
            metricsRouteDiscoveryStartTimes.erase(startIt);
        }
        auto candidateIt = metricsRouteDiscoveryCandidateCounts.find(target);
        if (candidateIt != metricsRouteDiscoveryCandidateCounts.end()) {
            metricsRouteCandidateCountSum += candidateIt->second;
            metricsRouteCandidateCountCount++;
            metricsRouteDiscoveryCandidateCounts.erase(candidateIt);
        }
        IRoute *selectedRoute = routingTable->findBestMatchingRoute(target);
        if (selectedRoute != nullptr) {
            metricsSelectedRouteHopCountSum += selectedRoute->getMetric();
            metricsSelectedRouteHopCountCount++;
        }
    }
    IRoute *selectedRouteForLog = routingTable->findBestMatchingRoute(target);
    if (selectedRouteForLog != nullptr && selectedRouteForLog->getSource() == this) {
        AodvRouteData *selectedData = dynamic_cast<AodvRouteData *>(selectedRouteForLog->getProtocolData());
        logRouteCauseEvent("ROUTE_DISCOVERY_SELECTED", target, selectedRouteForLog->getNextHopAsGeneric(), selectedRouteForLog->getMetric(),
                selectedData != nullptr && selectedData->isActive(), selectedData != nullptr ? selectedData->getLifeTime() : SIMTIME_ZERO, "completeRouteDiscovery");
    }

    auto lt = targetAddressToDelayedPackets.lower_bound(target);
    auto ut = targetAddressToDelayedPackets.upper_bound(target);

    // reinject the delayed datagrams
    for (auto it = lt; it != ut; it++) {
        Packet *datagram = it->second;
        const auto& networkHeader = getNetworkProtocolHeader(datagram);
        EV_DETAIL << "Sending queued datagram: source " << networkHeader->getSourceAddress() << ", destination " << networkHeader->getDestinationAddress() << endl;
        networkProtocol->reinjectQueuedDatagram(datagram);
    }

    // clear the multimap
    targetAddressToDelayedPackets.erase(lt, ut);

    // we have a route for the destination, thus we must cancel the WaitForRREPTimer events
    auto waitRREPIter = waitForRREPTimers.find(target);
    ASSERT(waitRREPIter != waitForRREPTimers.end());
    cancelAndDelete(waitRREPIter->second);
    waitForRREPTimers.erase(waitRREPIter);
}

void Aodv::sendGRREP(const Ptr<Rrep>& grrep, const L3Address& destAddr, unsigned int timeToLive)
{
    EV_INFO << "Sending gratuitous Route Reply to " << destAddr << endl;

    IRoute *destRoute = routingTable->findBestMatchingRoute(destAddr);
    if (destRoute == nullptr) {
        EV_WARN << "Cannot send gratuitous RREP to " << destAddr << ": no matching route exists anymore" << endl;
        return;
    }
    const L3Address& nextHop = destRoute->getNextHopAsGeneric();

    sendAODVPacket(grrep, nextHop, timeToLive, 0);
}

const Ptr<Rrep> Aodv::createHelloMessage()
{
    // called a Hello message, with the RREP
    // message fields set as follows:
    //
    //    Destination IP Address         The node's IP address.
    //
    //    Destination Sequence Number    The node's latest sequence number.
    //
    //    Hop Count                      0
    //
    //    Lifetime                       ALLOWED_HELLO_LOSS *HELLO_INTERVAL

    auto helloMessage = makeShared<Rrep>(); // TODO "AODV-HelloMsg");
    helloMessage->setPacketType(usingIpv6 ? RREP_IPv6 : RREP);
    helloMessage->setChunkLength(usingIpv6 ? B(44) : B(20));

    helloMessage->setDestAddr(getSelfIPAddress());
    helloMessage->setDestSeqNum(sequenceNum);
    helloMessage->setHopCount(0);
    helloMessage->setLifeTime(allowedHelloLoss * helloInterval);

    return helloMessage;
}

void Aodv::sendHelloMessagesIfNeeded()
{
    ASSERT(useHelloMessages);
    // Every HELLO_INTERVAL milliseconds, the node checks whether it has
    // sent a broadcast (e.g., a RREQ or an appropriate layer 2 message)
    // within the last HELLO_INTERVAL.  If it has not, it MAY broadcast
    // a RREP with TTL = 1

    // A node SHOULD only use hello messages if it is part of an
    // active route.
    bool hasActiveRoute = false;

    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        if (route->getSource() == this) {
            AodvRouteData *routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
            if (routeData->isActive()) {
                hasActiveRoute = true;
                break;
            }
        }
    }

    if (hasActiveRoute && (lastBroadcastTime == 0 || simTime() - lastBroadcastTime > helloInterval)) {
        EV_INFO << "It is hello time, broadcasting Hello Messages with TTL=1" << endl;
        auto helloMessage = createHelloMessage();
        sendAODVPacket(helloMessage, addressType->getBroadcastAddress(), 1, 0);

    }

    scheduleAfter(helloInterval - *periodicJitter, helloMsgTimer);
}

void Aodv::handleHelloMessage(const Ptr<Rrep>& helloMessage)
{
    const L3Address& helloOriginatorAddr = helloMessage->getDestAddr();
    IRoute *routeHelloOriginator = routingTable->findBestMatchingRoute(helloOriginatorAddr);

    // Whenever a node receives a Hello message from a neighbor, the node
    // SHOULD make sure that it has an active route to the neighbor, and
    // create one if necessary.  If a route already exists, then the
    // Lifetime for the route should be increased, if necessary, to be at
    // least ALLOWED_HELLO_LOSS * HELLO_INTERVAL.  The route to the
    // neighbor, if it exists, MUST subsequently contain the latest
    // Destination Sequence Number from the Hello message.  The current node
    // can now begin using this route to forward data packets.  Routes that
    // are created by hello messages and not used by any other active routes
    // will have empty precursor lists and would not trigger a RERR message
    // if the neighbor moves away and a neighbor timeout occurs.

    unsigned int latestDestSeqNum = helloMessage->getDestSeqNum();
    simtime_t newLifeTime = simTime() + allowedHelloLoss * helloInterval;

    if (!routeHelloOriginator || routeHelloOriginator->getSource() != this)
        createRoute(helloOriginatorAddr, helloOriginatorAddr, 1, true, latestDestSeqNum, true, newLifeTime);
    else {
        AodvRouteData *routeData = check_and_cast<AodvRouteData *>(routeHelloOriginator->getProtocolData());
        simtime_t lifeTime = routeData->getLifeTime();
        updateRoutingTable(routeHelloOriginator, helloOriginatorAddr, 1, true, latestDestSeqNum, true, std::max(lifeTime, newLifeTime));
    }

    // TODO This feature has not implemented yet.
    // A node MAY determine connectivity by listening for packets from its
    // set of neighbors.  If, within the past DELETE_PERIOD, it has received
    // a Hello message from a neighbor, and then for that neighbor does not
    // receive any packets (Hello messages or otherwise) for more than
    // ALLOWED_HELLO_LOSS * HELLO_INTERVAL milliseconds, the node SHOULD
    // assume that the link to this neighbor is currently lost.  When this
    // happens, the node SHOULD proceed as in Section 6.11.
}

void Aodv::expungeRoutes()
{
    bool routingTableChanged = false;
    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);
        if (route->getSource() == this) {
            AodvRouteData *routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
            ASSERT(routeData != nullptr);
            if (routeData->getLifeTime() <= simTime()) {
                if (routeData->isActive()) {
                    EV_DETAIL << "Route to " << route->getDestinationAsGeneric() << " expired and set to inactive. It will be deleted after DELETE_PERIOD time" << endl;
                    /*
                    appendAodvMetric("aodv_route_log.txt",
                            "time=" + simTime().str() +
                            ", node=" + std::string(getParentModule()->getFullName()) +
                            ", event=ROUTE_EXPIRED_INACTIVE, destination=" + route->getDestinationAsGeneric().str() +
                            ", nextHop=" + route->getNextHopAsGeneric().str());*/
                    // An expired routing table entry SHOULD NOT be expunged before
                    // (current_time + DELETE_PERIOD) (see section 6.11).  Otherwise, the
                    // soft state corresponding to the route (e.g., last known hop count)
                    // will be lost.
                    routeData->setIsActive(false);
                    routeData->setLifeTime(simTime() + deletePeriod);
                    diagnosisRouteExpireInactiveCount++;
                    logRouteCauseEvent("ROUTE_EXPIRE_INACTIVE", route->getDestinationAsGeneric(), route->getNextHopAsGeneric(), route->getMetric(), false, routeData->getLifeTime(), "expungeRoutes");
                    routingTableChanged = true;
                }
                else {
                    // Any routing table entry waiting for a RREP SHOULD NOT be expunged
                    // before (current_time + 2 * NET_TRAVERSAL_TIME).
                    if (hasOngoingRouteDiscovery(route->getDestinationAsGeneric())) {
                        EV_DETAIL << "Route to " << route->getDestinationAsGeneric() << " expired and is inactive, but we are waiting for a RREP to this destination, so we extend its lifetime with 2 * NET_TRAVERSAL_TIME" << endl;
                        routeData->setLifeTime(simTime() + 2 * netTraversalTime);
                    }
                    else {
                        EV_DETAIL << "Route to " << route->getDestinationAsGeneric() << " expired and is inactive and we are not expecting any RREP to this destination, so we delete this route" << endl;
                        /*
                        appendAodvMetric("aodv_route_log.txt",
                                "time=" + simTime().str() +
                                ", node=" + std::string(getParentModule()->getFullName()) +
                                ", event=ROUTE_DELETED, destination=" + route->getDestinationAsGeneric().str() +
                                ", nextHop=" + route->getNextHopAsGeneric().str());*/
                        logRouteCauseEvent("ROUTE_DELETE", route->getDestinationAsGeneric(), route->getNextHopAsGeneric(), route->getMetric(), false, routeData->getLifeTime(), "expungeRoutes");
                        diagnosisRouteDeleteCount++;
                        routingTable->deleteRoute(route);
                        routingTableChanged = true;
                    }
                }
            }
        }
    }
    if (routingTableChanged)
        logRoutingTableSnapshot("EXPUNGE_ROUTES");
    scheduleExpungeRoutes();
}

void Aodv::scheduleExpungeRoutes()
{
    simtime_t nextExpungeTime = SimTime::getMaxTime();
    for (int i = 0; i < routingTable->getNumRoutes(); i++) {
        IRoute *route = routingTable->getRoute(i);

        if (route->getSource() == this) {
            AodvRouteData *routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
            ASSERT(routeData != nullptr);

            if (routeData->getLifeTime() < nextExpungeTime)
                nextExpungeTime = routeData->getLifeTime();
        }
    }
    if (nextExpungeTime == SimTime::getMaxTime()) {
        if (expungeTimer->isScheduled())
            cancelEvent(expungeTimer);
    }
    else {
        if (!expungeTimer->isScheduled())
            scheduleAt(nextExpungeTime, expungeTimer);
        else {
            if (expungeTimer->getArrivalTime() != nextExpungeTime) {
                rescheduleAt(nextExpungeTime, expungeTimer);
            }
        }
    }
}

INetfilter::IHook::Result Aodv::datagramForwardHook(Packet *datagram)
{
    // TODO Implement: Actions After Reboot
    // If the node receives a data packet for some other destination, it SHOULD
    // broadcast a RERR as described in subsection 6.11 and MUST reset the waiting
    // timer to expire after current time plus DELETE_PERIOD.

    Enter_Method("datagramForwardHook");
    const auto& networkHeader = getNetworkProtocolHeader(datagram);
    const L3Address& destAddr = networkHeader->getDestinationAddress();
    const L3Address& sourceAddr = networkHeader->getSourceAddress();
    IRoute *ipSource = routingTable->findBestMatchingRoute(sourceAddr);

    if (destAddr.isBroadcast() || routingTable->isLocalAddress(destAddr) || destAddr.isMulticast()) {
        if (routingTable->isLocalAddress(destAddr) && ipSource && ipSource->getSource() == this)
            updateValidRouteLifeTime(ipSource->getNextHopAsGeneric(), simTime() + activeRouteTimeout);

        return ACCEPT;
    }

    // TODO IMPLEMENT: check if the datagram is a data packet or we take control packets as data packets

    IRoute *routeDest = routingTable->findBestMatchingRoute(destAddr);
    AodvRouteData *routeDestData = routeDest ? dynamic_cast<AodvRouteData *>(routeDest->getProtocolData()) : nullptr;

    // Each time a route is used to forward a data packet, its Active Route
    // Lifetime field of the source, destination and the next hop on the
    // path to the destination is updated to be no less than the current
    // time plus ACTIVE_ROUTE_TIMEOUT

    updateValidRouteLifeTime(sourceAddr, simTime() + activeRouteTimeout);
    updateValidRouteLifeTime(destAddr, simTime() + activeRouteTimeout);

    if (routeDest && routeDest->getSource() == this)
        updateValidRouteLifeTime(routeDest->getNextHopAsGeneric(), simTime() + activeRouteTimeout);

    // Since the route between each originator and destination pair is expected
    // to be symmetric, the Active Route Lifetime for the previous hop, along the
    // reverse path back to the IP source, is also updated to be no less than the
    // current time plus ACTIVE_ROUTE_TIMEOUT.

    if (ipSource && ipSource->getSource() == this)
        updateValidRouteLifeTime(ipSource->getNextHopAsGeneric(), simTime() + activeRouteTimeout);

    EV_INFO << "We can't forward datagram because we have no active route for " << destAddr << endl;
    if (routeDest && routeDestData && !routeDestData->isActive()) { // exists but is not active
        // A node initiates processing for a RERR message in three situations:
        // (ii)      if it gets a data packet destined to a node for which it
        //           does not have an active route and is not repairing (if
        //           using local repair)

        // TODO check if it is not repairing (if using local repair)

        // 1. The destination sequence number of this routing entry, if it
        // exists and is valid, is incremented for cases (i) and (ii) above,
        // and copied from the incoming RERR in case (iii) above.

        if (routeDestData->hasValidDestNum())
            routeDestData->setDestSeqNum(routeDestData->getDestSeqNum() + 1);

        // 2. The entry is invalidated by marking the route entry as invalid <- it is invalid

        // 3. The Lifetime field is updated to current time plus DELETE_PERIOD.
        //    Before this time, the entry SHOULD NOT be deleted.
        routeDestData->setLifeTime(simTime() + deletePeriod);
        diagnosisNoActiveRouteToForwardCount++;
        logRouteCauseEvent("NO_ACTIVE_ROUTE_TO_FORWARD", destAddr, routeDest->getNextHopAsGeneric(), routeDest->getMetric(), false, routeDestData->getLifeTime(), "datagramForwardHook");

        sendRERRWhenNoRouteToForward(destAddr);
    }
    else if (!routeDest || routeDest->getSource() != this) { // doesn't exist at all
        diagnosisNoRouteToForwardCount++;
        logRouteCauseEvent("NO_ROUTE_TO_FORWARD", destAddr, L3Address(), 0, false, SIMTIME_ZERO, "datagramForwardHook");
        sendRERRWhenNoRouteToForward(destAddr);
    }

    return ACCEPT;
}

void Aodv::sendRERRWhenNoRouteToForward(const L3Address& unreachableAddr)
{
    if (rerrCount >= rerrRatelimit) {
        EV_WARN << "A node should not generate more than RERR_RATELIMIT RERR messages per second. Canceling sending RERR" << endl;
        return;
    }
    std::vector<UnreachableNode> unreachableNodes;
    UnreachableNode node;
    node.addr = unreachableAddr;

    IRoute *unreachableRoute = routingTable->findBestMatchingRoute(unreachableAddr);
    AodvRouteData *unreachableRouteData = unreachableRoute ? dynamic_cast<AodvRouteData *>(unreachableRoute->getProtocolData()) : nullptr;

    if (unreachableRouteData && unreachableRouteData->hasValidDestNum())
        node.seqNum = unreachableRouteData->getDestSeqNum();
    else
        node.seqNum = 0;

    unreachableNodes.push_back(node);
    std::set<L3Address> precursorNodes;
    if (unreachableRouteData != nullptr) {
        for (const auto& precursorNode : unreachableRouteData->getPrecursorList())
            precursorNodes.insert(precursorNode);
    }
    logOriginatedRerr("NO_ROUTE_TO_FORWARD", unreachableNodes, precursorNodes);
    diagnosisRerrOriginatedCount++;
    logRouteCauseEvent("RERR_ORIGINATED", unreachableAddr, L3Address(), 0, false, SIMTIME_ZERO, "no_route_to_forward");
    auto rerr = createRERR(unreachableNodes);

    rerrCount++;
    EV_INFO << "Broadcasting Route Error message with TTL=1" << endl;
    sendAODVPacket(rerr, addressType->getBroadcastAddress(), 1, *jitterPar); // TODO unicast if there exists a route to the source
}

void Aodv::cancelRouteDiscovery(const L3Address& destAddr)
{
    ASSERT(hasOngoingRouteDiscovery(destAddr));
    metricsRouteDiscoveryStartTimes.erase(destAddr);
    metricsRouteDiscoveryCandidateCounts.erase(destAddr);
    auto lt = targetAddressToDelayedPackets.lower_bound(destAddr);
    auto ut = targetAddressToDelayedPackets.upper_bound(destAddr);
    for (auto it = lt; it != ut; it++)
        networkProtocol->dropQueuedDatagram(it->second);

    targetAddressToDelayedPackets.erase(lt, ut);

    auto waitRREPIter = waitForRREPTimers.find(destAddr);
    ASSERT(waitRREPIter != waitForRREPTimers.end());
    cancelAndDelete(waitRREPIter->second);
    waitForRREPTimers.erase(waitRREPIter);
}

bool Aodv::updateValidRouteLifeTime(const L3Address& destAddr, simtime_t lifetime)
{
    IRoute *route = routingTable->findBestMatchingRoute(destAddr);
    if (route && route->getSource() == this) {
        AodvRouteData *routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
        if (routeData->isActive()) {
            simtime_t newLifeTime = std::max(routeData->getLifeTime(), lifetime);
            EV_DETAIL << "Updating " << route << " lifetime to " << newLifeTime << endl;
            routeData->setLifeTime(newLifeTime);
            return true;
        }
    }
    return false;
}

const Ptr<RrepAck> Aodv::createRREPACK()
{
    auto rrepAck = makeShared<RrepAck>(); // TODO "AODV-RREPACK");
    rrepAck->setPacketType(usingIpv6 ? RREPACK_IPv6 : RREPACK);
    return rrepAck;
}

void Aodv::sendRREPACK(const Ptr<RrepAck>& rrepACK, const L3Address& destAddr)
{
    EV_INFO << "Sending Route Reply ACK to " << destAddr << endl;
    sendAODVPacket(rrepACK, destAddr, 100, 0);
}

void Aodv::handleRREPACK(const Ptr<const RrepAck>& rrepACK, const L3Address& neighborAddr)
{
    // Note that the RREP-ACK packet does not contain any information about
    // which RREP it is acknowledging.  The time at which the RREP-ACK is
    // received will likely come just after the time when the RREP was sent
    // with the 'A' bit.
    if (rrepAckTimer->isScheduled()) {
        EV_INFO << "RREP-ACK arrived from " << neighborAddr << endl;

        IRoute *route = routingTable->findBestMatchingRoute(neighborAddr);
        if (route && route->getSource() == this) {
            EV_DETAIL << "Marking route " << route << " as active" << endl;
            AodvRouteData *routeData = check_and_cast<AodvRouteData *>(route->getProtocolData());
            routeData->setIsActive(true);
            cancelEvent(rrepAckTimer);
        }
    }
}

void Aodv::handleRREPACKTimer()
{
    // when a node detects that its transmission of a RREP message has failed,
    // it remembers the next-hop of the failed RREP in a "blacklist" set.

    EV_INFO << "RREP-ACK didn't arrived within timeout. Adding " << failedNextHop << " to the blacklist" << endl;

    blacklist[failedNextHop] = simTime() + blacklistTimeout; // lifetime

    if (!blacklistTimer->isScheduled())
        scheduleAfter(blacklistTimeout, blacklistTimer);
}

void Aodv::handleBlackListTimer()
{
    simtime_t nextTime = SimTime::getMaxTime();

    for (auto it = blacklist.begin(); it != blacklist.end();) {
        auto current = it++;

        // Nodes are removed from the blacklist set after a BLACKLIST_TIMEOUT period
        if (current->second <= simTime()) {
            EV_DETAIL << "Blacklist lifetime has expired for " << current->first << " removing it from the blacklisted addresses" << endl;
            blacklist.erase(current);
        }
        else if (nextTime > current->second)
            nextTime = current->second;
    }

    if (nextTime != SimTime::getMaxTime())
        scheduleAt(nextTime, blacklistTimer);
}

Aodv::~Aodv()
{
    clearState();
    delete helloMsgTimer;
    delete expungeTimer;
    delete counterTimer;
    delete rrepAckTimer;
    delete blacklistTimer;
}

} // namespace aodv
} // namespace inet

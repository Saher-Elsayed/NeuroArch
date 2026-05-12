// NeuroArch UE5 BIM WebSocket Client
// Subscribes to bim_server.py; dispatches comfort state to material param collections.
// ComfortColorIndex: 0=Cold 1=Cool 2=Neutral 3=Warm 4=Hot
// Saturation = SNNConfidence (kappa_SNN, fades to gray at kappa=0.5)
#include "BIMWebSocketClient.h"
UBIMWebSocketClient::UBIMWebSocketClient()
    : ServerURL(TEXT("ws://localhost:8765")), bConnected(false) {}
void UBIMWebSocketClient::Connect() {}
void UBIMWebSocketClient::UpdateZone(const FString& ZoneGUID,
                                      const FString& ComfortClass,
                                      float SNNConfidence)
{
    float Idx = 2.0f;
    if (ComfortClass=="Cold")    Idx=0.f;
    else if (ComfortClass=="Cool")    Idx=1.f;
    else if (ComfortClass=="Warm")    Idx=3.f;
    else if (ComfortClass=="Hot")     Idx=4.f;
    // Call SetVectorParameterValue via UE5 Material Parameter Collection API
    // See Content/Blueprints/BP_ComfortHeatmap for full Blueprint implementation
}

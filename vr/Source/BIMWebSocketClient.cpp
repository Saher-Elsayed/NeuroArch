// BIMWebSocketClient.cpp
#include "BIMWebSocketClient.h"
#include "WebSocketsModule.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

UBIMWebSocketClient::UBIMWebSocketClient() {
    PrimaryComponentTick.bCanEverTick = false;
}

void UBIMWebSocketClient::BeginPlay() {
    Super::BeginPlay();
    Connect();
}

void UBIMWebSocketClient::EndPlay(const EEndPlayReason::Type Reason) {
    Disconnect();
    Super::EndPlay(Reason);
}

void UBIMWebSocketClient::Connect() {
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
        FModuleManager::Get().LoadModule("WebSockets");

    WebSocket = FWebSocketsModule::Get().CreateWebSocket(ServerUrl, TEXT(""));
    WebSocket->OnConnected().AddLambda([this]() {
        bConnected = true;
        OnConnectionStatus.Broadcast(true);
        UE_LOG(LogTemp, Log, TEXT("NeuroArch BIM WebSocket connected: %s"), *ServerUrl);
    });
    WebSocket->OnConnectionError().AddLambda([this](const FString& Err) {
        bConnected = false;
        OnConnectionStatus.Broadcast(false);
        UE_LOG(LogTemp, Warning, TEXT("BIM WS error: %s"), *Err);
        ScheduleReconnect();
    });
    WebSocket->OnClosed().AddLambda([this](int32 Code, const FString& Reason, bool bRemote) {
        bConnected = false;
        OnConnectionStatus.Broadcast(false);
        if (bRemote) ScheduleReconnect();
    });
    WebSocket->OnMessage().AddLambda([this](const FString& Msg) { HandleMessage(Msg); });
    WebSocket->Connect();
}

void UBIMWebSocketClient::Disconnect() {
    if (WebSocket.IsValid() && bConnected)
        WebSocket->Close();
}

void UBIMWebSocketClient::Subscribe(const FString& ZoneId) {
    if (!bConnected) return;
    TSharedPtr<FJsonObject> Req = MakeShareable(new FJsonObject());
    Req->SetStringField(TEXT("cmd"), TEXT("subscribe"));
    Req->SetStringField(TEXT("zone"), ZoneId);
    FString ReqStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&ReqStr);
    FJsonSerializer::Serialize(Req.ToSharedRef(), Writer);
    WebSocket->Send(ReqStr);
}

void UBIMWebSocketClient::HandleMessage(const FString& Msg) {
    double RxTime = FPlatformTime::Seconds() * 1000.0;
    TSharedPtr<FJsonObject> Json;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Msg);
    if (!FJsonSerializer::Deserialize(Reader, Json) || !Json.IsValid()) return;
    FComfortUpdate Update = ParseDelta(Json);
    Update.LatencyMs = (float)(RxTime - LastMessageTime.GetTicks() / 10000.0);
    LatencyHistory.Add(Update.LatencyMs);
    if (LatencyHistory.Num() > 1000) LatencyHistory.RemoveAt(0);
    LastMessageTime = FDateTime::UtcNow();
    OnComfortUpdate.Broadcast(Update);
}

FComfortUpdate UBIMWebSocketClient::ParseDelta(const TSharedPtr<FJsonObject>& Json) {
    FComfortUpdate U;
    Json->TryGetStringField(TEXT("zone"), U.ZoneId);
    const TSharedPtr<FJsonObject>* Delta;
    if (Json->TryGetObjectField(TEXT("delta"), Delta)) {
        (*Delta)->TryGetStringField(TEXT("comfort_class"), U.ComfortClass);
        (*Delta)->TryGetNumberField(TEXT("confidence"), U.Confidence);
        (*Delta)->TryGetNumberField(TEXT("pmv"), U.PMVValue);
        (*Delta)->TryGetNumberField(TEXT("ppd"), U.PPDValue);
        (*Delta)->TryGetNumberField(TEXT("supply_temp"), U.SupplyTempSet);
    }
    return U;
}

float UBIMWebSocketClient::GetMeanLatencyMs() const {
    if (LatencyHistory.IsEmpty()) return 0.f;
    float Sum = 0.f;
    for (float L : LatencyHistory) Sum += L;
    return Sum / LatencyHistory.Num();
}

void UBIMWebSocketClient::ScheduleReconnect() {
    FTimerHandle Handle;
    GetWorld()->GetTimerManager().SetTimer(Handle, [this]() { Connect(); },
                                           ReconnectDelaySeconds, false);
}

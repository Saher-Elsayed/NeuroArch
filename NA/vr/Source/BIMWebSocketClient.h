// BIMWebSocketClient.h — Unreal Engine 5.3 BIM WebSocket Client
// Receives NeuroArch comfort and HVAC delta updates over WebSocket
// and applies them to Unreal BIM model actors.
//
// Usage: Attach to an AActor. Set ServerUrl and call Connect().
// Subscribe to OnComfortUpdate delegate for UI updates.

#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "IWebSocket.h"
#include "BIMWebSocketClient.generated.h"

USTRUCT(BlueprintType)
struct FComfortUpdate {
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) FString ZoneId;
    UPROPERTY(BlueprintReadOnly) FString ComfortClass;  // Cold/Cool/Neutral/Warm/Hot
    UPROPERTY(BlueprintReadOnly) float   Confidence;
    UPROPERTY(BlueprintReadOnly) float   PMVValue;
    UPROPERTY(BlueprintReadOnly) float   PPDValue;
    UPROPERTY(BlueprintReadOnly) float   SupplyTempSet;
    UPROPERTY(BlueprintReadOnly) float   LatencyMs;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnComfortUpdate, const FComfortUpdate&, Update);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnConnectionStatus, bool, bConnected);

UCLASS(ClassGroup=(NeuroArch), meta=(BlueprintSpawnableComponent))
class NEUROARCH_API UBIMWebSocketClient : public UActorComponent {
    GENERATED_BODY()

public:
    UBIMWebSocketClient();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="NeuroArch")
    FString ServerUrl = TEXT("ws://localhost:8765");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="NeuroArch")
    float ReconnectDelaySeconds = 2.0f;

    UPROPERTY(BlueprintAssignable, Category="NeuroArch")
    FOnComfortUpdate OnComfortUpdate;

    UPROPERTY(BlueprintAssignable, Category="NeuroArch")
    FOnConnectionStatus OnConnectionStatus;

    UFUNCTION(BlueprintCallable, Category="NeuroArch")
    void Connect();

    UFUNCTION(BlueprintCallable, Category="NeuroArch")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category="NeuroArch")
    void Subscribe(const FString& ZoneId);

    UFUNCTION(BlueprintPure, Category="NeuroArch")
    bool IsConnected() const { return bConnected; }

    UFUNCTION(BlueprintPure, Category="NeuroArch")
    float GetMeanLatencyMs() const;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;

private:
    TSharedPtr<IWebSocket> WebSocket;
    bool bConnected = false;
    TArray<float> LatencyHistory;
    FDateTime LastMessageTime;

    void HandleMessage(const FString& MessageString);
    FComfortUpdate ParseDelta(const TSharedPtr<FJsonObject>& Json);
    void ScheduleReconnect();
};

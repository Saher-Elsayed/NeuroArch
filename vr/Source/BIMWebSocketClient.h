// NeuroArch UE5 BIM WebSocket Client header
// Paper: Section V-B, Stage 3 (live data binding)
#pragma once
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "BIMWebSocketClient.generated.h"

UCLASS()
class NEUROARCHVR_API UBIMWebSocketClient : public UObject {
    GENERATED_BODY()
public:
    UBIMWebSocketClient();
    UFUNCTION(BlueprintCallable) void Connect();
    UFUNCTION(BlueprintCallable)
    void UpdateZone(const FString& ZoneGUID, const FString& ComfortClass, float SNNConfidence);
    UPROPERTY(EditAnywhere) FString ServerURL;
private:
    bool bConnected;
};

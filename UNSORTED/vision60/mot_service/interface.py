from dataclasses import dataclass


@dataclass
class SingleMOTDetectioknResults:
    frame_no: int
    id: int
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    class_: int
    confidence: int

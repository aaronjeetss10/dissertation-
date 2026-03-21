# DJI Neo SAR Benchmark — Datasheet

## Motivation

**Purpose:** Evaluate action recognition methods (MViTv2-S, TMS) at realistic
SAR drone altitudes with controlled ground-truth annotations.

**Creators:** [Author Name], [University], 2026

**Funding:** University dissertation project (self-funded)

## Composition

| Property | Value |
|---|---|
| Total clips | 48 |
| Successfully processed | 48 |
| Actions | falling, crawling, lying_down, running, waving, climbing, walking, stumbling |
| Altitudes | 50m, 75m, 100m |
| Takes per combo | 2 |
| Actors | 3–4 volunteers |
| Drone | DJI Neo |
| Resolution | 1920×1080 (4K scaled) |
| Frame rate | 30fps (processed at 5fps) |
| Duration per clip | 10–20 seconds |

## Collection Process

- **Location:** Open field / terrain, UK
- **Weather:** Daylight, clear/overcast
- **Protocol:** Each actor performs each action twice at each altitude
- **Safety:** No real emergencies simulated; actors briefed on safety
- **Ethics:** Verbal consent from all participants; no identifiable faces at altitude

## Preprocessing

- Clips named: `{action}_{altitude}m_{take}.mp4`
- Processed through SARTriage pipeline (YOLO11n → ByteTrack → 6 streams)
- Manual annotations via `annotate.py` for ground-truth validation

## Uses

- Primary: Validate TMS vs MViTv2-S accuracy at controlled altitudes
- Secondary: Test AAI crossover hypothesis with real altitude data
- Tertiary: Measure detection recall vs altitude

## Distribution

- Available on request for academic use
- Not publicly released (contains participant video)

## Maintenance

- Maintained as part of dissertation repository
- Contact: [Author email]

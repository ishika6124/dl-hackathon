import scipy.io as sio
import numpy as np
import sys

# Force UTF-8 output to avoid Windows cp1252 encoding errors
sys.stdout.reconfigure(encoding='utf-8')

data = sio.loadmat('train.mat')

print("=== TOP LEVEL KEYS ===")
for k in data:
    if k.startswith('__'): continue
    val = data[k]
    print(f"  '{k}' : type={type(val).__name__}, shape={getattr(val,'shape','?')}, value={str(val.flat[0] if hasattr(val,'flat') else val)[:60]}")

print("\n=== DS SENSOR STRUCT ===")
ds = data['DS'][0, 0]   # unwrap the struct
print(f"DS fields: {ds.dtype.names}")

print("\n=== DS FIELDS DETAIL ===")
for field in ds.dtype.names:
    val = ds[field]
    print(f"\n  '{field}':")
    print(f"    dtype={val.dtype}, shape={val.shape}")
    if field == 'rawData':
        print(f"    -> vibration signal matrix: {val.shape[0]} measurements x {val.shape[1]} samples each")
        print(f"    -> first signal, first 5 values: {val[0, :5]}")
    elif field == 'label':
        unique, counts = np.unique(val.flatten(), return_counts=True)
        print(f"    -> labels: {dict(zip(unique.tolist(), counts.tolist()))}")
    elif field == 'RPM':
        r = val.flatten()
        print(f"    -> RPM values: min={r.min():.1f}, max={r.max():.1f}, first 3: {r[:3]}")
    elif field == 'samplingRate':
        print(f"    -> sampling rate: {val.flat[0]} Hz")
    elif field == 'faultFrequencies':
        print(f"    -> NESTED struct - unwrapping...")
        ff = val[0, 0]
        print(f"       fields: {ff.dtype.names}")
        for ff_field in ff.dtype.names:
            mult = float(ff[ff_field].flat[0])
            print(f"       '{ff_field}' = {mult:.6f}  (shaft frequency multiplier)")

print("\n=== HOW FEATURES ARE EXTRACTED FROM THIS STRUCTURE ===")
print("""
ACCESS PATH:
  data['DS'][0,0]                         -> sensor struct
  data['DS'][0,0]['rawData']              -> shape (N, 4096)  -> raw signal
  data['DS'][0,0]['label']                -> shape (N, 1)     -> class label (0/1/2/3)
  data['DS'][0,0]['RPM']                  -> shape (N, 1)     -> shaft speed
  data['DS'][0,0]['samplingRate']         -> scalar           -> Hz
  data['DS'][0,0]['faultFrequencies'][0,0]['BPFIMultiple'] -> scalar multiplier
  data['DS'][0,0]['faultFrequencies'][0,0]['BPFOMultiple'] -> scalar multiplier
  data['DS'][0,0]['faultFrequencies'][0,0]['BPFMultiple']  -> scalar multiplier
  data['DS'][0,0]['faultFrequencies'][0,0]['FTFMultiple']  -> scalar multiplier

FEATURE EXTRACTION (preprocess.py):
  1. RAW SIGNAL (4096,)   -> directly from rawData row
  2. STATISTICAL (18,)    -> mean, std, rms, kurtosis, skewness, crest, etc. on raw signal
  3. ENVELOPE/FAULT (4,)  -> Hilbert transform -> envelope -> FFT
                             amplitude at: mult * (RPM/60) Hz for each fault type
                             BPFImult * shaft_hz  -> BPFI amplitude
                             BPFOmult * shaft_hz  -> BPFO amplitude
                             BPFmult  * shaft_hz  -> BPF amplitude
                             FTFmult  * shaft_hz  -> FTF amplitude
  4. METADATA (8,)        -> RPM, samplingRate, folder_id, label_encoded, etc.
""")

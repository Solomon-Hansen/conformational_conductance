from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns

px = 1 / plt.rcParams["figure.dpi"]
plt.rcParams["figure.figsize"] = [1848 * px, 965 * px]
c = [
    "#007fff",  # Blue
    "#ff3616",  # Red
    "#138d75",  # Green
    "#7d3c98",  # Purple
    "#fbea6a",  # Yellow
]
sns.set(
    style="ticks",
    rc={
        "font.family": "Arial",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 3,
    },
    font_scale=2.5,
    palette=sns.color_palette("Set2"),
)

CONDUCTANCE_LABEL = r"Conductance (log$_{10}$(G/G$_0$))"
ALPHABET = "ABCDEFGHIJKLMNOPQ"
RNG_STATE = 43
FONTSIZE = 48
UPPER_THRESHOLD = -0.01
LOWER_THRESHOLD = -8
FS = 40000

millisecond_xaxis = np.arange(0, 20000, dtype=float)
millisecond_xaxis /= 40


def prep_traces(traces, piezo_pull):
    new_traces = []
    new_piezos = []
    count = 0
    for t in tqdm(traces):
        p = piezo_pull.copy()
        num_pts = len(t)
        start_idx = int(num_pts * 0.95)
        end_idx = int(num_pts * 0.96)
        avg_val = np.mean(t[start_idx:end_idx])
        noise = avg_val
        t -= avg_val
        zero_cutoff = 8 * 10**-5
        avg_val = np.mean(t[0 : int(num_pts * 0.01)])
        if (noise < zero_cutoff) & (np.abs(avg_val) > 1.02):
            # if (np.abs(avg_val) > 1.02):
            t = t[: int(len(t) * 0.90)]
            p = p[: int(len(t) * 0.90)]
            t = np.log10(t)
            indices = np.argwhere(t < -0.5).ravel()
            if len(indices) < 1:
                continue
            t = t[indices[3] :]
            p = p[indices[3] :]
            if len(np.argwhere(t[: int(len(t) * 0.10)] > -0.5)) > 10:
                continue
            indices = np.argwhere(t > -6.7).ravel()
            t = t[indices]

            if len(t) < 16:
                count += 1
                continue

            new_traces.append(t)
            new_piezos.append(p)
        else:
            count += 1

    print(f"Traces skipped: {count = }")
    new_traces = np.array(new_traces, dtype=object)
    new_piezos = np.array(new_piezos, dtype=object)
    return new_traces, new_piezos


extension = np.load(
    "../piezoBiasWave_FlickerNoise_ABiPh_pullout.npz", allow_pickle=True
)["traces"]
piezo_pull = extension[75000 // 100]
piezo_pull = np.array(piezo_pull, dtype=np.float32)
del extension

traces = np.load("../../STM-BJ/FN_Astilb_conductance.npz", allow_pickle=True)["traces"]
prepped_traces, prepped_piezos = prep_traces(traces, piezo_pull)


def find_inflections(piezo):
    double_grad = np.gradient(np.gradient(piezo))
    double_grad /= np.std(double_grad)
    inflections = np.argwhere(np.abs(double_grad) > 20).ravel()
    inflections = inflections[
        np.argwhere((inflections[1:] - inflections[:-1]) > 10)
    ].ravel()
    return inflections


new_prepped = []
for trace, piezo in tqdm(
    zip(prepped_traces, prepped_piezos), total=len(prepped_traces)
):
    try:
        inflections = find_inflections(piezo)
        if len(inflections) == 2:
            new_prepped.append(trace[inflections[0] : inflections[1]])
        if len(inflections) == 1:
            if inflections[0] > 1000:
                new_prepped.append(trace[: inflections[0]])
    except:
        continue
prepped_traces = new_prepped


def extract_by_mean(traces, select_mean, std):
    prime_traces = []
    upper_bound = select_mean + std
    lower_bound = select_mean - std
    for trace in tqdm(traces):
        if len(trace) > 3000:
            m = np.mean(trace[400:-1000])
            if (m < upper_bound) & (m > lower_bound):
                prime_traces.append(trace[:-256])
    return prime_traces


def get_scaling_exponent(welch_psds, means, grid):
    result = []
    logged_means = np.log10(means)
    for i in grid:
        normalized = np.log10(welch_psds / (means**i))
        correlation = pearsonr(logged_means, normalized)[0]
        result.append(correlation)
    result = np.array(result)
    argindex = np.argsort(np.abs(result))[0]
    scaling_exponent = grid[argindex]
    return scaling_exponent


def analyze_traces(
    forward_shift: int,
    welch_points: int,
    start_idx_shift: int,
    last_idx_shift: int,
    std: float,
    mean: float,
):
    welch_psds = []
    means = []
    selected_traces = extract_by_mean(prepped_traces, mean, std=std)

    for idx, trace in enumerate(selected_traces):
        subset_trace = trace[forward_shift : forward_shift + welch_points]
        if len(subset_trace) >= welch_points:
            exponentiated = 10**subset_trace
            f, Pxx_den = signal.welch(
                exponentiated,
                nperseg=welch_points,
                window="boxcar",
                scaling="density",
                fs=FS,
            )

            if idx == 0:
                start_idx = np.argsort(np.abs(100 - f))[0] + start_idx_shift
                start_idx = max(0, start_idx)
                last_idx = start_idx * 10 + last_idx_shift

            means.append(np.mean(exponentiated))
            welch_psds.append(np.sum(Pxx_den[start_idx:last_idx]))
    return np.array(means), np.array(welch_psds)


STEP_SIZE = 0.020


def analyze_sensitivity(nsamples: int):
    rng = np.random.default_rng()
    forward_shift_parameters = [
        rng.integers(low=1000, high=3500, size=nsamples, endpoint=True)
    ]
    welch_points_parameters = [
        rng.integers(low=1000, high=3500, size=nsamples, endpoint=True)
    ]
    start_idx_shift_parameters = [
        rng.integers(low=-2, high=4, size=nsamples, endpoint=True)
    ]
    last_idx_shift_parameters = [
        rng.integers(low=-15, high=35, size=nsamples, endpoint=True)
    ]
    std_parameters = [rng.uniform(0.40, 0.60, size=nsamples)]
    parameter_bag = np.concatenate(
        (
            forward_shift_parameters,
            welch_points_parameters,
            start_idx_shift_parameters,
            last_idx_shift_parameters,
            std_parameters,
        ),
        axis=0,
    ).T

    scaling_exponents = []
    for forward_shift, welch_points, start_idx, last_idx, std in tqdm(parameter_bag):
        try:
            means, welch_psds = analyze_traces(
                forward_shift=int(forward_shift),
                welch_points=int(welch_points),
                start_idx_shift=int(start_idx),
                last_idx_shift=int(last_idx),
                std=std,
                mean=-2.9,  # high g state
                # mean=-4.4,  # low g state
            )
        except Exception as e:
            print(e)
            print(f"{forward_shift = }")
            print(f"{welch_points = }")
            print(f"{start_idx = }")
            print(f"{last_idx = }")
            print(f"{std = }")
            break
        try:
            grid = np.arange(0.75, 3.0, STEP_SIZE)
            scaling_exponent = get_scaling_exponent(
                welch_psds=welch_psds, means=means, grid=grid
            )
            scaling_exponents.append(scaling_exponent)
        except Exception as e:
            print(f"exception occured: {e}")
            continue
    return scaling_exponents


scaling_exponents = analyze_sensitivity(3000)

print(f"{np.mean(scaling_exponents):.6}")
print(f"{np.std(scaling_exponents) + STEP_SIZE:.6}")


def cumulative_std(arr):
    """
    Compute the cumulative standard deviation along a given axis.

    Parameters:
    arr (numpy.ndarray): Input array.
    axis (int): Axis along which the cumulative standard deviation is computed.

    Returns:
    numpy.ndarray: Array of the cumulative standard deviations.
    """
    arr = np.array(arr)
    cum_std = np.zeros_like(arr, dtype=float)
    for i in range(1, arr.shape[0] + 1):
        cum_std[i - 1] = np.std(arr[:i], ddof=0)
    return cum_std


counts, binedges = np.histogram(scaling_exponents, bins=128)
np.savetxt("./scaling_exponents.csv", scaling_exponents, delimiter=",")
binedges = (binedges[1:] + binedges[:-1]) / 2
plt.plot(binedges, counts)
plt.xlabel("Scaling Exponent")
plt.ylabel("Counts")
plt.xlim(np.min(binedges), np.max(binedges))
plt.ylim(
    0,
)
plt.savefig("./scaling_exponents.png")

# plt.plot(range(len(scaling_exponents)), cumulative_std(scaling_exponents))
# plt.scatter(len(scaling_exponents) -1, np.std(scaling_exponents), marker="x")

plt.plot(np.cumsum(scaling_exponents) / np.arange(1, len(scaling_exponents) + 1))
plt.ylabel("Scaling Exponent")
plt.xlabel("# iteration")
plt.xlim(
    0,
)
plt.title("Cumulated sum of scaling exponents")
plt.savefig("./cumulated_scaling_exponents.png")

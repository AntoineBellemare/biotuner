import concurrent.futures
import more_itertools

def harmonic_fit(peaks, n_harm=10, bounds=1, function="mult", div_mode="div", n_common_harms=5):
    from itertools import combinations
    peak_bands = np.arange(len(peaks))
    if function == "mult":
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == "div":
        multi_harmonics, x = EEG_harmonics_div(peaks, n_harm, mode=div_mode)
    elif function == "exp":
        multi_harmonics = []
        for h in range(n_harm + 1):
            h += 1
            multi_harmonics.append([i**h for i in peaks])
        multi_harmonics = np.array(multi_harmonics)
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)
    list_peaks = list(combinations(peak_bands, 2))
    harm_temp = []
    matching_positions = []
    harmonics_pos = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(compareLists, multi_harmonics[i[0]], multi_harmonics[i[1]], bounds) for i in list_peaks]
        for f in concurrent.futures.as_completed(results):
            harms, harm_pos, matching_pos, _ = f.result()
            harm_temp.append(harms)
            harmonics_pos.append(harm_pos)
            matching_positions.extend(matching_pos)

    matching_positions = [list(i) for i in matching_positions]
    harm_fit = np.array(harm_temp, dtype=object).squeeze()
    harmonics_pos = reduce(lambda x, y: x + y, harmonics_pos)
    most_common_harmonics = [
        h
        for h, h_count in Counter(harmonics_pos).most_common(n_common_harms)
        if h_count > 1
    ]
    harmonics_pos = np.sort(np.unique(harmonics_pos))
    if len(peak_bands) > 2:
        harm_fit = more_itertools.flatten(harm_fit)
        harm_fit = np.round(harm_fit, 3)
        harm_fit = np.unique(harm_fit)
    return harm_fit, harmonics_pos, most_common_harmonics, matching_positions

import numpy as np
import random
from sklearn.feature_selection import mutual_info_regression
from scipy.special import expit
from scipy import optimize
import n_sphere


########################################################################################################################
# Self Censoring Sigmoid
########################################################################################################################
def simulate_nan_mnar_sigmoid(
        data, cols, missing_ratios, missing_funcs, strict=False, corr_type='all', seed=1002031, beta_corr=None
):
    mask = np.zeros(data.shape, dtype=bool)

    # add missing for each column
    for col in cols:

        if isinstance(missing_ratios, dict):
            missing_ratio = missing_ratios[col]
        elif isinstance(missing_ratios, list):
            missing_ratio = missing_ratios[cols.index(col)]
        else:
            missing_ratio = missing_ratios

        if isinstance(missing_funcs, dict):
            missing_func = missing_funcs[col]
        elif isinstance(missing_funcs, list):
            missing_func = missing_funcs[cols.index(col)]
        else:
            missing_func = missing_funcs

        # set the seed
        seed = (seed + 1203941) % (2 ^ 32 - 1)

        # missing is associated with column itself
        if corr_type == 'self':
            data_corr = data[:, col]
        elif corr_type == 'others':
            data_corr = data[:, [i for i in cols if i != col]]
        elif corr_type == 'all':
            data_corr = data
        elif corr_type.startswith('allk'):
            np.random.seed(seed)
            k = max(int(float(corr_type.split('allk')[-1]) * data.shape[1]), 1)
            mi = np.corrcoef(data, rowvar=False)[col]
            mi_idx = np.argsort(mi)[::-1][:k + 1]
            data_corr = data[:, mi_idx]
            if k == 1:
                data_corr = data_corr.reshape(-1, 1)
        else:
            raise NotImplementedError

        #################################################################################
        # pick coefficients and mask missing valuesn
        #################################################################################
        mask = mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed, beta_corr)

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms

########################################################################################################################
# Self Censoring Quantile
########################################################################################################################
def simulate_nan_mnar_quantile(
		data, cols, missing_ratios, missing_funcs='left', strict=True, seed=201030
):

	# find the columns that are not to be adding missing values
	mask = np.zeros(data.shape, dtype=bool)

	for col in cols:

		if isinstance(missing_ratios, dict):
			missing_ratio = missing_ratios[col]
		elif isinstance(missing_ratios, list):
			missing_ratio = missing_ratios[cols.index(col)]
		else:
			missing_ratio = missing_ratios

		if isinstance(missing_funcs, dict):
			missing_func = missing_funcs[col]
		elif isinstance(missing_funcs, list):
			missing_func = missing_funcs[cols.index(col)]
		else:
			missing_func = missing_funcs

		# set the seed
		seed = (seed + 10087651) % (2 ** 32 - 1)
		random.seed(seed)
		np.random.seed(seed)

		data_corr = data[:, col]

		# find the quantile of the most correlated column
		if missing_func == 'random':
			missing_func = random.choice(['left', 'right', 'mid', 'tail'])

		# get mask based on quantile
		mask = mask_mar_quantile(mask, col, data_corr, missing_ratio, missing_func, strict, seed)

	# assign the missing values
	data_ms = data.copy()
	data_ms[mask] = np.nan

	return data_ms


########################################################################################################################
# Helper Functions
########################################################################################################################
def generate_param_vector(d, main_strength=30, direction='up'):
    # sampling a vector from unit sphere
    if direction == 'up':
        theta_1 = np.random.uniform(0, main_strength)
    else:
        theta_1 = np.random.uniform(180 - main_strength, 180)
    if d < 3:
        return n_sphere.convert_rectangular([1, np.radians(theta_1)])
    else:
        theta_others = []
        for i in range(d - 3):
            theta_others.append(np.random.uniform(0, 90))
        theta_last = np.random.uniform(0, 360)
        thetas = [theta_1] + theta_others + [theta_last]
        thetas = [np.radians(theta) for theta in thetas]
        sphere_coords = [1] + thetas
        return n_sphere.convert_rectangular(sphere_coords)

def mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed, beta_corr):

    np.random.seed(seed)
    random.seed(seed)

    #################################################################################
    # pick coefficients
    #################################################################################
    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    if isinstance(missing_ratio, dict):
        missing_ratio = missing_ratio[col]
    else:
        missing_ratio = missing_ratio

    # copy data and do min-max normalization
    data_copy = data_corr.copy()

    if data_copy.ndim == 1:
        data_copy = data_copy.reshape(-1, 1)

    data_copy = (data_copy - data_copy.min(0, keepdims=True)) / (
            data_copy.max(0, keepdims=True) - data_copy.min(0, keepdims=True) + 1e-5)
    data_copy = (data_copy - data_copy.mean(0, keepdims=True)) / \
                (data_copy.std(0, keepdims=True) + 1e-5)

    # if beta_corr is True, then the coefficients are calculated based on the correlation
    if beta_corr is None:
        coeffs = np.random.rand(data_copy.shape[1], 1)
    elif beta_corr == 'b1':
        coeffs = np.random.rand(data_copy.shape[1], 1)
        corr_ri = np.corrcoef(data_copy, rowvar=False)[0].reshape(-1, 1)
        low_bound = 0.1
        np.where(corr_ri < low_bound, low_bound, corr_ri)
        coeffs = coeffs * corr_ri
        coeffs[0] = 1.0
    elif beta_corr == 'b2':
        coeffs = np.random.rand(data_copy.shape[1], 1)
        ri = np.corrcoef(data_copy, rowvar=False)[0].reshape(-1, 1)
        coeffs[0] = 1
        coeffs = coeffs / coeffs.max()
        coeffs = coeffs * np.sign(ri)
    elif beta_corr == 'sphere':
        if missing_func == 'left':
            coeffs = generate_param_vector(data_copy.shape[1], main_strength=30, direction='up')
        else:
            coeffs = generate_param_vector(data_copy.shape[1], main_strength=30, direction='up')
    elif beta_corr == 'sphere2':
            np.random.seed(col)
            random_vector = np.random.randn(data_copy.shape[1])
            unit_vector = random_vector / np.linalg.norm(random_vector)
            if unit_vector[0] < 0:
                unit_vector = -unit_vector
            coeffs = unit_vector.reshape(-1, 1)
    else:
        raise NotImplementedError

    # Wx = data_copy @ coeffs
    # coeffs /= (np.std(Wx, 0, keepdims=True) + 1e-3)

    def f1(x: np.ndarray) -> np.ndarray:
        if missing_func == 'left':
            return expit(-np.dot(data_copy, coeffs) + x).mean().item() - missing_ratio
        elif missing_func == 'right':
            return expit(np.dot(data_copy, coeffs) + x).mean().item() - missing_ratio
        elif missing_func == 'mid':
            return expit(-np.absolute(np.dot(data_copy, coeffs)) + 0.75 + x).mean().item() - missing_ratio
        elif missing_func == 'tail':
            return expit(np.absolute(np.dot(data_copy, coeffs)) - 0.75 + x).mean().item() - missing_ratio
        else:
            raise NotImplementedError

    def f2(x: np.ndarray) -> np.ndarray:
        if missing_func == 'left':
            return np.quantile(expit(-np.dot(data_copy, coeffs) + x), 1-missing_ratio).item() - missing_ratio
        elif missing_func == 'right':
            return np.quantile(expit(np.dot(data_copy, coeffs) + x), 1-missing_ratio).item() - missing_ratio
        elif missing_func == 'mid':
            return np.quantile(expit(np.absolute(np.dot(data_copy, coeffs)) - 0.75 + x), 1-missing_ratio).item() - missing_ratio
        elif missing_func == 'tail':
            return np.quantile(expit(-np.absolute(np.dot(data_copy, coeffs)) + 0.75 + x), 1-missing_ratio).item() - missing_ratio
        else:
            raise NotImplementedError

    if strict is False:
        intercept = optimize.bisect(f1, -50, 50)
        if missing_func == 'left':
            ps = expit(-np.dot(data_copy, coeffs) + intercept)
        elif missing_func == 'right':
            ps = expit(np.dot(data_copy, coeffs) + intercept)
        elif missing_func == 'mid':
            ps = expit(-np.absolute(np.dot(data_copy, coeffs)) + 0.75 + intercept)
        elif missing_func == 'tail':
            ps = expit(np.absolute(np.dot(data_copy, coeffs)) - 0.75 + intercept)
        else:
            raise NotImplementedError

        ps = ps.flatten()
        ber = np.random.rand(data_copy.shape[0])
        mask[:, col] = ber < ps

    else:
        intercept = optimize.bisect(f2, -50, 50)
        if missing_func == 'left':
            ps = expit(-np.dot(data_copy, coeffs) + intercept)
        elif missing_func == 'right':
            ps = expit(np.dot(data_copy, coeffs) + intercept)
        elif missing_func == 'mid':
            ps = expit(-np.absolute(np.dot(data_copy, coeffs)) + 0.75 + intercept)
        elif missing_func == 'tail':
            ps = expit(np.absolute(np.dot(data_copy, coeffs)) - 0.75 + intercept)
        else:
            raise NotImplementedError

        ps = ps.flatten()
        missing_N_upper = int((missing_ratio + 0.02) * data_copy.shape[0])
        missing_N_lower = int((missing_ratio - 0.02) * data_copy.shape[0])
        if (ps >= missing_ratio).sum() > missing_N_upper or (ps >= missing_ratio).sum() < missing_N_lower:
            idx = np.argsort(ps)[::-1]
            mask[idx[:int(data_copy.shape[0]*missing_ratio)], col] = True
        else:
            mask[:, col] = ps >= missing_ratio


    #print(f"Column {col} missing ratio: {mask[:, col].sum() / mask.shape[0]} Expected: {missing_ratio}")

    return mask

def mask_mar_quantile(mask, col, data_corr, missing_ratio, missing_func, strict, seed):

	if strict:
		total_missing = int(missing_ratio * data_corr.shape[0])
		sorted_values = np.sort(data_corr)
		if missing_func == 'left':
			q = sorted_values[int(missing_ratio * data_corr.shape[0]) - 1]
			indices = np.where(data_corr < q)[0]

			if len(indices) < total_missing:
				end_indices = np.where(
					data_corr == q
				)[0]
				add_up_indices = np.random.choice(
					end_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(
					indices, size=total_missing, replace=False
				)
			else:
				na_indices = indices
		elif missing_func == 'right':
			q = sorted_values[int((1 - missing_ratio) * data_corr.shape[0])]
			indices = np.where(data_corr > q)[0]
			if len(indices) < total_missing:
				start_indices = np.where(
					data_corr == q
				)[0]
				add_up_indices = np.random.choice(
					start_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(
					indices, size=total_missing, replace=False
				)
			else:
				na_indices = indices
		elif missing_func == 'mid':
			q0 = sorted_values[int((1 - missing_ratio) / 2 * data_corr.shape[0])]
			q1 = sorted_values[int((1 + missing_ratio) / 2 * data_corr.shape[0]) - 1]
			indices = np.where(
				(data_corr > q0) & (data_corr < q1)
			)[0]
			if len(indices) < total_missing:
				end_indices_q0 = np.where(data_corr == q0)[0]
				end_indices_q1 = np.where(data_corr == q1)[0]
				end_indices = np.union1d(
					end_indices_q0, end_indices_q1
				)
				add_up_indices = np.random.choice(
					end_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(
					indices, size=total_missing, replace=False
				)
			else:
				na_indices = indices
		elif missing_func == 'tail':
			q0 = sorted_values[int(missing_ratio / 2 * data_corr.shape[0])]
			q1 = sorted_values[int((1 - missing_ratio / 2) * data_corr.shape[0]) - 1]
			indices = np.where(
				(data_corr < q0) | (data_corr > q1)
			)[0]

			if len(indices) < total_missing:
				end_indices_q0 = np.where(data_corr == q0)[0]
				end_indices_q1 = np.where(data_corr == q1)[0]
				print(missing_func, q0, q1, len(indices), total_missing, len(data_corr), len(end_indices_q0),
					  len(end_indices_q1))
				end_indices = np.union1d(
					end_indices_q0, end_indices_q1
				)
				add_up_indices = np.random.choice(
					end_indices, size=total_missing - len(indices), replace=False
				)
				na_indices = np.concatenate((indices, add_up_indices))
			elif len(indices) > total_missing:
				na_indices = np.random.choice(
					indices, size=total_missing, replace=False
				)
			else:
				na_indices = indices
		else:
			raise NotImplementedError
	else:
		if missing_func == 'left':
			q0 = 0
			q1 = 0.5 if missing_ratio <= 0.5 else missing_ratio
		elif missing_func == 'right':
			q0 = 0.5 if missing_ratio <= 0.5 else 1 - missing_ratio
			q1 = 1
		elif missing_func == 'mid' or missing_func == 'tail':
			q0 = 0.25 if missing_ratio <= 0.5 else 0.5 - missing_ratio / 2
			q1 = 0.75 if missing_ratio <= 0.5 else 0.5 + missing_ratio / 2
		else:
			raise NotImplementedError

		sorted_values = np.sort(data_corr)
		q0 = sorted_values[int(q0 * data_corr.shape[0])]
		q1 = sorted_values[int(q1 * data_corr.shape[0]) - 1]

		if missing_func != 'tail':
			indices = np.where(
				(data_corr >= q0) & (data_corr <= q1)
			)[0]
		else:
			indices = np.where(
				(data_corr <= q0) | (data_corr >= q1)
			)[0]
		np.random.seed(seed)
		na_indices = np.random.choice(
			indices, size=int(missing_ratio * data_corr.shape[0]), replace=False
		)

	mask[na_indices, col] = True

	return mask

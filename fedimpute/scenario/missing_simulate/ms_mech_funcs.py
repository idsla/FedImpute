import random
from scipy.special import expit
import numpy as np
from n_sphere import n_sphere
from scipy import optimize


def generate_param_vector(d, main_strength=30, direction='up', rng = np.random.default_rng(10020)):
    # sampling a vector from unit sphere
    if direction == 'up':
        theta_1 = rng.uniform(0, main_strength)
    else:
        theta_1 = rng.uniform(180 - main_strength, 180)

    if d < 3:
        return n_sphere.convert_rectangular([1, np.radians(theta_1)])
    else:
        theta_others = []
        for i in range(d - 3):
            theta_others.append(rng.uniform(0, 90))
        theta_last = rng.uniform(0, 360)
        thetas = [theta_1] + theta_others + [theta_last]
        thetas = [np.radians(theta) for theta in thetas]
        sphere_coords = [1] + thetas
        return n_sphere.convert_rectangular(sphere_coords)


def mask_sigmoid(
        mask: np.ndarray, col: int, data_corr: np.ndarray, missing_ratio: float, missing_func: str, strict: bool,
        mechanism: str, beta_corr: str, rng: np.random.Generator
):
    # np.random.seed(seed)
    # random.seed(seed)

    if mechanism not in ['mar', 'mnar']:
        raise ValueError('mechanism should be one of mar or mnar')

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

    ####################################################################################################################
    # How to set betas in logistic function
    if mechanism == 'mnar':
        if beta_corr == 'self':  # self masking MNAR missingness
            coeffs = np.zeros((data_copy.shape[1], 1))
            coeffs[0] = 1.0
        elif beta_corr == 'sphere':  # randomly to axis cone of feature itself for MNAR missingness
            coeffs = generate_param_vector(data_copy.shape[1], main_strength=30, direction='up')
        elif beta_corr == 'randu':  # randomly set beta coefficients for MNAR missingness - (-1, 1)
            coeffs = rng.random((data_copy.shape[1], 1))
            coeffs = coeffs / np.linalg.norm(coeffs)
            coeffs[0] = 1.0
        else:
            raise ValueError('Unknown beta_corr, it should be one of self, sphere, random for MNAR mechanism.')
    elif mechanism == 'mar':
        if beta_corr == 'fixed':
            coeffs = np.ones((data_copy.shape[1], 1))
            coeffs = coeffs / np.linalg.norm(coeffs)
            coeffs = coeffs.reshape(-1, 1)
        elif beta_corr == 'randu':  # randomly set beta coefficients - mix of mnar and mar missingness
            random_vector = rng.random((data_copy.shape[1], 1)) * 2 - 1  # random uniform between -1 and 1
            unit_vector = random_vector / np.linalg.norm(random_vector)
            coeffs = unit_vector.reshape(-1, 1)
        elif beta_corr == 'randn':  # randomly set beta coefficients - mix of mnar and mar missingness
            random_vector = rng.standard_normal(data_copy.shape[1])
            unit_vector = random_vector / np.linalg.norm(random_vector)
            coeffs = unit_vector.reshape(-1, 1)
        else:
            raise ValueError('Unknown beta_corr, it should be one of random_uniform, random_normal for MAR mechanism.')
    else:
        raise ValueError('mechanism should be one of mar or mnar')

    # elif beta_corr == 'b1':
    #     coeffs = np.random.rand(data_copy.shape[1], 1)
    #     corr_ri = np.corrcoef(data_copy, rowvar=False)[0].reshape(-1, 1)
    #     low_bound = 0.1
    #     np.where(corr_ri < low_bound, low_bound, corr_ri)
    #     coeffs = coeffs * corr_ri
    #     coeffs[0] = 1.0
    # elif beta_corr == 'b2':
    #     coeffs = np.random.rand(data_copy.shape[1], 1)
    #     ri = np.corrcoef(data_copy, rowvar=False)[0].reshape(-1, 1)
    #     coeffs[0] = 1
    #     coeffs = coeffs / coeffs.max()
    #     coeffs = coeffs * np.sign(ri)
    # elif beta_corr == 'sphere2':    # randomly select one vector
    #     np.random.seed(col)
    #     random_vector = np.random.randn(data_copy.shape[1])
    #     unit_vector = random_vector / np.linalg.norm(random_vector)
    #     if unit_vector[0] < 0:
    #         unit_vector = -unit_vector
    #     coeffs = unit_vector.reshape(-1, 1)
    # else:
    #     raise ValueError('Unknown beta_corr, it should be one of random, b1, b2, sphere, sphere2.')

    # Wx = data_copy @ coeffs
    # coeffs = np.random.rand(data_copy.shape[1], 1)
    # wss = (Wx) / (np.std(Wx, 0, keepdims=True) + 1e-3)

    ####################################################################################################################
    # optimization function for sigmoid function missing ratio
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
            return np.quantile(expit(-np.dot(data_copy, coeffs) + x), 1 - missing_ratio).item() - missing_ratio
        elif missing_func == 'right':
            return np.quantile(expit(np.dot(data_copy, coeffs) + x), 1 - missing_ratio).item() - missing_ratio
        elif missing_func == 'mid':
            return np.quantile(expit(np.absolute(np.dot(data_copy, coeffs)) - 0.75 + x),
                               1 - missing_ratio).item() - missing_ratio
        elif missing_func == 'tail':
            return np.quantile(expit(-np.absolute(np.dot(data_copy, coeffs)) + 0.75 + x),
                               1 - missing_ratio).item() - missing_ratio
        else:
            raise NotImplementedError

    ####################################################################################################################
    # calculate intercept for missing ratios
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
            mask[idx[:int(data_copy.shape[0] * missing_ratio)], col] = True
        else:
            #print(mask.shape, ps.shape, missing_ratio)
            mask[:, col] = ps >= missing_ratio

    return mask


def mask_quantile(
        mask: np.array, col: int, data_corr: np.array, missing_ratio: float, missing_func: str, strict: bool,
        rng: np.random.Generator
) -> np.array:
    """
    Mask missing values based on quantile
    :param mask: missing mask
    :param col: column to add missing values
    :param data_corr: data the missing value associated with
    :param ms_ratio: missing ratio
    :param missing_func: missing function
    :param strict: whether to strictly add missing values
    :param rng: random generator
    :return: updated mask
    """

    # np.random.seed(seed)
    # random.seed(seed)
    N = data_corr.shape[0]

    if strict:
        total_missing = int(missing_ratio * N)
        sorted_values = np.sort(data_corr)
        if missing_func == 'left':
            q = sorted_values[int(missing_ratio * N) - 1]
            indices = np.where(data_corr < q)[0]

            if len(indices) < total_missing:
                end_indices = np.where(data_corr == q)[0]
                add_up_indices = rng.choice(end_indices, size=total_missing - len(indices), replace=False)
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = rng.choice(indices, size=total_missing, replace=False)
            else:
                na_indices = indices
        elif missing_func == 'right':
            q = sorted_values[int((1 - missing_ratio) * N)]
            indices = np.where(data_corr > q)[0]
            if len(indices) < total_missing:
                start_indices = np.where(data_corr == q)[0]
                add_up_indices = rng.choice(start_indices, size=total_missing - len(indices), replace=False)
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = rng.choice(indices, size=total_missing, replace=False)
            else:
                na_indices = indices
        elif missing_func == 'mid':
            q0 = sorted_values[int((1 - missing_ratio) / 2 * N)]
            q1 = sorted_values[int((1 + missing_ratio) / 2 * N) - 1]
            indices = np.where((data_corr > q0) & (data_corr < q1))[0]
            if len(indices) < total_missing:
                end_indices_q0 = np.where(data_corr == q0)[0]
                end_indices_q1 = np.where(data_corr == q1)[0]
                end_indices = np.union1d(end_indices_q0, end_indices_q1)
                add_up_indices = rng.choice(end_indices, size=total_missing - len(indices), replace=False)
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = rng.choice(indices, size=total_missing, replace=False)
            else:
                na_indices = indices
        elif missing_func == 'tail':
            q0 = sorted_values[int(missing_ratio / 2 * N)]
            q1 = sorted_values[int((1 - missing_ratio / 2) * N) - 1]
            indices = np.where((data_corr < q0) | (data_corr > q1))[0]

            if len(indices) < total_missing:
                end_indices_q0 = np.where(data_corr == q0)[0]
                end_indices_q1 = np.where(data_corr == q1)[0]
                end_indices = np.union1d(end_indices_q0, end_indices_q1)
                add_up_indices = rng.choice(end_indices, size=total_missing - len(indices), replace=False)
                na_indices = np.concatenate((indices, add_up_indices))
            elif len(indices) > total_missing:
                na_indices = rng.choice(indices, size=total_missing, replace=False)
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
        q0 = sorted_values[int(q0 * N)]
        q1 = sorted_values[int(q1 * N) - 1]

        if missing_func != 'tail':
            indices = np.where((data_corr >= q0) & (data_corr <= q1))[0]
        else:
            indices = np.where((data_corr <= q0) | (data_corr >= q1))[0]
        # np.random.seed(seed)
        na_indices = rng.choice(indices, size=int(missing_ratio * N), replace=False)

    mask[na_indices, col] = True

    return mask

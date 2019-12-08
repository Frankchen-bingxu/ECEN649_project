import glob
import os
import pickle
from datetime import datetime
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from joblib import Parallel, delayed
from numba import jit
from sklearn.metrics import *
import warnings

warnings.filterwarnings('ignore')

WINDOW_SIZE = 19
Size = NamedTuple('Size', [('height', int), ('width', int)])
Location = NamedTuple('Location', [('top', int), ('left', int)])
ThresholdPolarity = NamedTuple('ThresholdPolarity', [('threshold', float), ('polarity', float)])
ClassifierResult = NamedTuple('ClassifierResult',
                              [('threshold', float), ('polarity', int), ('classification_error', float),
                               ('classifier', Callable[[np.ndarray], float])])
WeakClassifier = NamedTuple('WeakClassifier', [('threshold', float), ('polarity', int), ('alpha', float),
                                               ('classifier', Callable[[np.ndarray], float])])


def to_float_array(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32) / 255.


def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values * 255.))


def gamma(values: np.ndarray, coeff: float = 2.2) -> np.ndarray:
    return values ** (1. / coeff)


def gleam(values: np.ndarray) -> np.ndarray:
    return np.sum(gamma(values), axis=2) / values.shape[2]


def to_integral(img: np.ndarray) -> np.ndarray:
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]


def open_face(path: str, resize: bool = True) -> Image.Image:
    img = Image.open(path)
    img = to_image(gamma(to_float_array(img)))
    return img.convert('L')


dataset_path = 'C:/dataset/'

faces_dir = os.path.join(dataset_path, 'train/face')
face_image_files = glob.glob(os.path.join(faces_dir, '**', '*.pgm'), recursive=True)
nofaces_dir = os.path.join(dataset_path, 'train/non-face')
noface_image_files = glob.glob(os.path.join(nofaces_dir, '**', '*.pgm'), recursive=True)
print('The trainface loding...:')
print(str(len(face_image_files)) + ' face loaded')
print('The train non-face loding...:')
print(str(len(noface_image_files)) + ' nonface loaded\n')
test_faces_dir = os.path.join(dataset_path, 'test/face')
test_face_image_files = glob.glob(os.path.join(test_faces_dir, '**', '*.pgm'), recursive=True)
test_nofaces_dir = os.path.join(dataset_path, 'test/non-face')
test_noface_image_files = glob.glob(os.path.join(test_nofaces_dir, '**', '*.pgm'), recursive=True)


class Box:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.coords_x = [x, x + width, x, x + width]
        self.coords_y = [y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1]

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))


class Feature:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        self.coords_x = [x, x + hw, x, x + hw,
                         x + hw, x + width, x + hw, x + width]
        self.coords_y = [y, y, y + height, y + height,
                         y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hh = height // 2
        self.coords_x = [x, x + width, x, x + width,
                         x, x + width, x, x + width]
        self.coords_y = [y, y, y + hh, y + hh,
                         y + hh, y + hh, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1]


class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        self.coords_x = [x, x + tw, x, x + tw,
                         x + tw, x + 2 * tw, x + tw, x + 2 * tw,
                         x + 2 * tw, x + width, x + 2 * tw, x + width]
        self.coords_y = [y, y, y + height, y + height,
                         y, y, y + height, y + height,
                         y, y, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 3
        self.coords_x = [x, x + width, x, x + width,
                         x, x + width, x, x + width,
                         x, x + width, x, x + width]
        self.coords_y = [y, y, y + th, y + th,
                         y + th, y + th, y + 2 * th, y + 2 * th,
                         y + 2 * th, y + 2 * th, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature4(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.coords_x = [x, x + hw, x, x + hw,  # upper row
                         x + hw, x + width, x + hw, x + width,
                         x, x + hw, x, x + hw,  # lower row
                         x + hw, x + width, x + hw, x + width]
        self.coords_y = [y, y, y + hh, y + hh,  # upper row
                         y, y, y + hh, y + hh,
                         y + hh, y + hh, y + height, y + height,  # lower row
                         y + hh, y + hh, y + height, y + height]
        self.coeffs = [1, -1, -1, 1,  # upper row
                       -1, 1, 1, -1,
                       -1, 1, 1, -1,  # lower row
                       1, -1, -1, 1]


def possible_position(size: int, window_size: int = WINDOW_SIZE) -> Iterable[int]:
    return range(0, window_size - size + 1)


def possible_locations(base_shape: Size, window_size: int = WINDOW_SIZE) -> Iterable[Location]:
    return (Location(left=x, top=y)
            for x in possible_position(base_shape.width, window_size)
            for y in possible_position(base_shape.height, window_size))


def possible_shapes(base_shape: Size, window_size: int = WINDOW_SIZE) -> Iterable[Size]:
    base_height = base_shape.height
    base_width = base_shape.width
    return (Size(height=height, width=width)
            for width in range(base_width, window_size + 1, base_width)
            for height in range(base_height, window_size + 1, base_height))


feature2h = list(Feature2h(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=1, width=2), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature2v = list(Feature2v(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=2, width=1), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature3h = list(Feature3h(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=1, width=3), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature3v = list(Feature3v(location.left, location.top, shape.width, shape.height)
                 for shape in possible_shapes(Size(height=3, width=1), WINDOW_SIZE)
                 for location in possible_locations(shape, WINDOW_SIZE))

feature4 = list(Feature4(location.left, location.top, shape.width, shape.height)
                for shape in possible_shapes(Size(height=2, width=2), WINDOW_SIZE)
                for location in possible_locations(shape, WINDOW_SIZE))

features = feature2h + feature2v + feature3h + feature3v + feature4

print(f'Number of feature2h features: {len(feature2h)}')
print(f'Number of feature2v features: {len(feature2v)}')
print(f'Number of feature3h features: {len(feature3h)}')
print(f'Number of feature3v features: {len(feature3v)}')
print(f'Number of feature4 features:  {len(feature4)}')
print(f'Total number of features:     {len(features)}')

def train_data():
    xs = []
    xs.extend([to_float_array(open_face(f)) for f in face_image_files])
    xs.extend([to_float_array(open_face(f)) for f in noface_image_files])
    p = len(face_image_files)
    n = len(noface_image_files)
    ys = np.hstack([np.ones((p,)), np.zeros((n,))])
    return np.array(xs), ys


def test_data():
    xs = []
    xs.extend([to_float_array(open_face(f)) for f in test_face_image_files])
    xs.extend([to_float_array(open_face(f)) for f in test_noface_image_files])
    p = len(test_face_image_files)
    n = len(test_noface_image_files)
    ys = np.hstack([np.ones((p,)), np.zeros((n,))])
    return np.array(xs), ys


xs, _ = train_data()

train_mean = xs.mean()
train_std = xs.std()
del xs
del _

test_xs, _ = test_data()

test_mean = test_xs.mean()
test_std = test_xs.std()

del test_xs
del _


def normalize(im: np.ndarray, mean: float = train_mean, std: float = train_std) -> np.ndarray:
    return (im - mean) / std


def train_data_normalized(mean: float = train_mean, std: float = train_std) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = train_data()
    xs = normalize(xs, mean, std)
    return xs, ys


def test_data_normalized(mean: float = test_mean, std: float = test_std) -> Tuple[np.ndarray, np.ndarray]:
    test_xs, test_ys = test_data()
    test_xs = normalize(test_xs, mean, std)
    return test_xs, test_ys


####Adaptive Boosting

@jit
def weak_classifier(x: np.ndarray, f: Feature, polarity: float, theta: float) -> float:
    # return 1. if (polarity * f(x)) < (polarity * theta) else 0.
    return (np.sign((polarity * theta) - (polarity * f(x))) + 1) // 2


@jit
def run_weak_classifier(x: np.ndarray, c: WeakClassifier) -> float:
    return weak_classifier(x=x, f=c.classifier, polarity=c.polarity, theta=c.threshold)


@jit
def strong_classifier(x: np.ndarray, weak_classifiers: List[WeakClassifier]) -> int:
    sum_hypotheses = 0.
    sum_alphas = 0.
    for c in weak_classifiers:
        sum_hypotheses += c.alpha * run_weak_classifier(x, c)
        sum_alphas += c.alpha
    return 1 if (sum_hypotheses >= .5 * sum_alphas) else 0


def normalize_weights(w: np.ndarray) -> np.ndarray:
    return w / w.sum()


@jit
def build_running_sums(ys: np.ndarray, ws: np.ndarray) -> Tuple[float, float, List[float], List[float]]:
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.
    s_minuses, s_pluses = [], []

    for y, w in zip(ys, ws):
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)
    return t_minus, t_plus, s_minuses, s_pluses


@jit
def find_best_threshold(zs: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                        s_pluses: List[float]) -> ThresholdPolarity:
    min_e = float('inf')
    min_z, polarity = 0, 0
    for z, s_m, s_p in zip(zs, s_minuses, s_pluses):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        if error_1 < min_e:
            min_e = error_1
            min_z = z
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            min_z = z
            polarity = 1
    return ThresholdPolarity(threshold=min_z, polarity=polarity)


def determine_threshold_polarity(ys: np.ndarray, ws: np.ndarray, zs: np.ndarray) -> ThresholdPolarity:
    # Sort according to score
    p = np.argsort(zs)
    zs, ys, ws = zs[p], ys[p], ws[p]

    # Determine the best threshold: build running sums
    t_minus, t_plus, s_minuses, s_pluses = build_running_sums(ys, ws)

    # Determine the best threshold: select optimal threshold.
    return find_best_threshold(zs, t_minus, t_plus, s_minuses, s_pluses)


def apply_feature(f: Feature, xis: np.ndarray, ys: np.ndarray, ws: np.ndarray,
                  parallel: Optional[Parallel] = None) -> ClassifierResult:
    if parallel is None:
        parallel = Parallel(n_jobs=-1, backend='threading')

    # Determine all feature values
    zs = np.array(parallel(delayed(f)(x) for x in xis))

    # Determine the best threshold
    result = determine_threshold_polarity(ys, ws, zs)

    # Determine the classification error
    classification_error = 0.
    for x, y, w in zip(xis[500:1999], ys[500:1999], ws[500:1999]):
        h = weak_classifier(x, f, result.polarity, result.threshold)
        classification_error += w * np.abs(h - y)


    return ClassifierResult(threshold=result.threshold, polarity=result.polarity,
                            classification_error=classification_error, classifier=f)


STATUS_EVERY = 10
def build_weak_classifiers(prefix: str, num_features: int, xis: np.ndarray, ys: np.ndarray, features: List[Feature],
                           ws: Optional[np.ndarray] = None) -> Tuple[List[WeakClassifier], List[float]]:
    if ws is None:
        m = len(ys[ys < .5])  # number of negative example
        l = len(ys[ys > .5])  # number of positive examples

        # Initialize the weights
        ws = np.zeros_like(ys)
        ws[ys < .5] = 1. / (2. * m)
        ws[ys > .5] = 1. / (2. * l)

    # Keep track of the history of the example weights.
    w_history = [ws]

    total_start_time = datetime.now()
    with Parallel(n_jobs=-1, backend='threading') as parallel:
        weak_classifiers = []  # type: List[WeakClassifier]
        for t in range(num_features):
            print(f'Building weak classifier {t + 1}/{num_features} ...')
            start_time = datetime.now()

            # Normalize the weights
            ws = normalize_weights(ws)

            status_counter = STATUS_EVERY

            # Select best weak classifier for this round
            best = ClassifierResult(polarity=0, threshold=0, classification_error=float('inf'), classifier=None)
            for i, f in enumerate(features):
                status_counter -= 1
                improved = False
                # Python runs singlethreaded. To speed things up,
                # we're only anticipating every other feature, give or take.
                if KEEP_PROBABILITY < 1.:
                    skip_probability = np.random.random()
                    if skip_probability > KEEP_PROBABILITY:
                        continue

                result = apply_feature(f, xis, ys, ws, parallel)
                if result.classification_error < best.classification_error:
                    improved = True
                    best = result

                # Print status every couple of iterations.
                if improved or status_counter == 0:
                    current_time = datetime.now()
                    duration = current_time - start_time
                    total_duration = current_time - total_start_time
                    status_counter = STATUS_EVERY
                    if improved:
                        print(
                            f't={t + 1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s in this stage) {i + 1}/{len(features)} {100 * i / len(features):.2f}% evaluated. Classification error improved to {best.classification_error:.5f} using {str(best.classifier)} ,the best threshold is {best.threshold:.5f}...')
                    else:
                        print(
                            f't={t + 1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s in this stage) {i + 1}/{len(features)} {100 * i / len(features):.2f}% evaluated.')
            # After the best classifier was found, determine alpha
            beta = best.classification_error / (1 - best.classification_error)
            alpha = np.log(1. / beta)

            # Build the weak classifier
            classifier = WeakClassifier(threshold=best.threshold, polarity=best.polarity, classifier=best.classifier,
                                        alpha=alpha)

            # Update the weights for misclassified examples
            for i, (x, y) in enumerate(zip(xis, ys)):
                h = run_weak_classifier(x, classifier)
                e = np.abs(h - y)
                ws[i] = ws[i] * np.power(beta, 1 - e)

            # Register this weak classifier
            weak_classifiers.append(classifier)
            w_history.append(ws)

            pickle.dump(classifier, open(f'{prefix}-weak-learner-{t + 1}-of-{num_features}.pickle', 'wb'))

    print(f'Done building {num_features} weak classifiers.')
    return weak_classifiers, w_history


KEEP_PROBABILITY = 1. / 4.


PredictionStats = NamedTuple('PredictionStats', [('tn', int), ('fp', int), ('fn', int), ('tp', int)])


def prediction_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, PredictionStats]:
    c = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c.ravel()
    return c, PredictionStats(tn=tn, fp=fp, fn=fn, tp=tp)


xs, ys = train_data_normalized()
xis = np.array([to_integral(x) for x in xs])

##train model
assert xis.shape[1:3] == (WINDOW_SIZE + 1, WINDOW_SIZE + 1), xis.shape
weak_classifiers, w_history = build_weak_classifiers('1st', 1, xis, ys, features)


##test model
test_xs, test_ys = test_data_normalized()
test_xis = np.array([to_integral(x) for x in test_xs])

##Performance of the first weak classifier
# ys_1 = np.array([run_weak_classifier(x, weak_classifiers[0]) for x in test_xis])
# c, s = prediction_stats(test_ys, ys_1)
#
# sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'{weak_classifiers[0].classifier} alone')
#
#
# print(f'1st stage, classifier0, Precision {s.tp / (s.tp + s.fp):.2f}, recall {s.tp / (s.tp + s.fn):.2f}, false positive rate {s.fp / (s.fp + s.tn):.2f}, false negative rate {s.fn / (s.tp + s.fn):.2f}.')
#
# ##the performance of the second weak classifier, conditioned on the first one
# ys_2 = np.array([run_weak_classifier(x, weak_classifiers[1]) for x in test_xis])
# c, s = prediction_stats(test_ys, ys_2)
#
# sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'{weak_classifiers[1].classifier} alone')
#
# print(
#     f'1st stage, classifier1, Precision {s.tp / (s.tp + s.fp):.2f}, recall {s.tp / (s.tp + s.fn):.2f}, false positive rate {s.fp / (s.fp + s.tn):.2f}, false negative rate {s.fn / (s.tp + s.fn):.2f}.')


##Output of the combined classifier
ys_strong = np.array([strong_classifier(x, weak_classifiers) for x in test_xis])
c, s = prediction_stats(test_ys, ys_strong)

sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
            xticklabels=['Predicted negative', 'Predicted positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion matrix for the strong classifier')
plt.show()

print(f'{weak_classifiers[0].classifier} alone')
# print(f'{weak_classifiers[1].classifier} alone')
# print(f'{weak_classifiers[2].classifier} alone')
# print(f'{weak_classifiers[3].classifier} alone')
# print(f'{weak_classifiers[4].classifier} alone')


print(
    f'Combine 1 weak classifiers in the 1st stage, Precision {s.tp / (s.tp + s.fp):.2f}, recall {s.tp / (s.tp + s.fn):.2f}, false positive rate {s.fp / (s.fp + s.tn):.2f}, false negative rate {s.fn / (s.tp + s.fn):.2f}.')


##Second stage weak classifier
# weak_classifiers_2, w_history_2 = build_weak_classifiers('2nd', 2, xis, ys, features)
# weak_classifiers_2
#
# ys_strong = np.array([strong_classifier(x, weak_classifiers_2) for x in test_xis])
# c, s = prediction_stats(test_ys, ys_strong)
#
# sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'Confusion matrix for the strong classifier (stage 2)');
# plt.show()
# print(
#     f'Combine weak classifiers in the 2nd stage, Precision {s.tp / (s.tp + s.fp):.2f}, recall {s.tp / (s.tp + s.fn):.2f}, false positive rate {s.fp / (s.fp + s.tn):.2f}, false negative rate {s.fn / (s.tp + s.fn):.2f}.')
# #
# ##Third stage weak classifier
# weak_classifiers_3, w_history_3 = build_weak_classifiers('3rd', 25, xis, ys, features)
# weak_classifiers_3
#
# ys_strong = np.array([strong_classifier(x, weak_classifiers_3) for x in test_xis])
# c, s = prediction_stats(test_ys, ys_strong)
#
# sns.heatmap(c / c.sum(), cmap='YlGnBu', annot=True, square=True, fmt='.1%',
#             xticklabels=['Predicted negative', 'Predicted positive'],
#             yticklabels=['Negative', 'Positive'])
# plt.title(f'Confusion matrix for the strong classifier (stage 3)');
#
# print(
#     f'Combine weak classifiers in the 3rd stage, Precision {s.tp / (s.tp + s.fp):.2f}, recall {s.tp / (s.tp + s.fn):.2f}, false positive rate {s.fp / (s.fp + s.tn):.2f}, false negative rate {s.fn / (s.tp + s.fn):.2f}.')

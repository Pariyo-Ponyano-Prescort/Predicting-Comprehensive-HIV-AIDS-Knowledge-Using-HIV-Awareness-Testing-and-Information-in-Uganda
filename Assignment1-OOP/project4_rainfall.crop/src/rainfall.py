import numpy as np
from scipy.signal import find_peaks

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


class Region:
    """Store and analyse monthly rainfall for a region."""

    def __init__(self, name: str, rainfall: list[float]):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Region name cannot be empty.")

        values = np.asarray(rainfall, dtype=float)

        if values.shape != (12,):
            raise ValueError("Enter exactly 12 monthly values.")

        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("Rainfall must be non-negative and finite.")

        self.name = name
        self.rainfall = values.copy()

    def annual_total(self) -> float:
        """Return total annual rainfall in mm."""
        return float(np.sum(self.rainfall))

    def mean(self) -> float:
        """Return mean monthly rainfall in mm."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> str:
        """Return the wettest month."""
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Return the driest month."""
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Return coefficient of variation as a percentage."""
        average = self.mean()
        if average == 0:
           return float("nan")
        return float(np.std(self.rainfall) / average * 100)

    def rainy_seasons(self) -> list[str]:
        """Return months identified as prominent rainfall peaks."""
        peaks, _ = find_peaks(self.rainfall, prominence=25)
        return [MONTHS[int(i)] for i in peaks]

    def season_pattern(self) -> str:
        """Classify the number of detected rainfall peaks."""
        count = len(self.rainy_seasons())

        if count >= 2:
            return "Bimodal"
        if count == 1:
            return "Unimodal"
        return "No clear peak"
    
    @staticmethod
    def detect_rainy_seasons(rainfall, prominence=25):
        """Detect rainy-season peaks in 12 monthly rainfall values."""
        rainfall = np.asarray(rainfall, dtype=float)
        if rainfall.shape != (12,):
           raise ValueError("Rainfall data must contain exactly 12 monthly values.")
        peaks, properties = find_peaks(rainfall, prominence=prominence)
        return {
        "peak_indices": peaks,
        "peak_values": rainfall[peaks],
        "properties": properties,
        "peak_months": [MONTHS[int(i)] for i in peaks]
        }



class CropRule:
    """Classify monthly rainfall using crop screening thresholds."""

    def __init__(
        self, name: str, minimum: float, maximum: float
    ):
        if not name.strip():
            raise ValueError("Crop name cannot be empty.")
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("Thresholds must be finite.")
        if minimum < 0 or maximum <= minimum:
            raise ValueError("Invalid rainfall thresholds.")

        self.name = name
        self.minimum = minimum
        self.maximum = maximum

    def classify(self, rainfall: float) -> str:
        """Return a crop suitability label for monthly rainfall."""
        if not np.isfinite(rainfall) or rainfall < 0:
            raise ValueError("Rainfall must be finite and non-negative.")

        if rainfall < self.minimum:
            return "Drought risk"
        if rainfall > self.maximum:
            return "Waterlogging risk"
        return f"Good for {self.name}"


def cosine_similarity(a, b) -> float:
    """Calculate cosine similarity using NumPy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Vectors must have equal lengths and be 1-D.")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Vectors must contain finite values.")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        raise ValueError("Cannot compare a zero vector.")

    return float(np.dot(a, b) / denominator)


def compare_regions(regions: dict):
    """Return cosine, Pearson and Euclidean comparison matrices."""
    names = list(regions)
    n = len(names)

    cosine_matrix = np.eye(n)
    pearson_matrix = np.eye(n)
    euclidean_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = regions[names[i]].rainfall
            b = regions[names[j]].rainfall

            cosine_matrix[i, j] = cosine_matrix[j, i] = (
                cosine_similarity(a, b)
            )

            if np.std(a) == 0 or np.std(b) == 0:
                correlation = np.nan
            else:
                correlation = np.corrcoef(a, b)[0, 1]

            pearson_matrix[i, j] = pearson_matrix[j, i] = correlation
            euclidean_matrix[i, j] = euclidean_matrix[j, i] = (
                np.linalg.norm(a - b)
            )

    return {
        "Cosine similarity": cosine_matrix,
        "Pearson correlation": pearson_matrix,
        "Euclidean distance": euclidean_matrix,
    }
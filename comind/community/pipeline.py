from dataclasses import dataclass
from pathlib import Path
from .dataset import Dataset
from comind.config import Config
from comind.utils import MetricValue, WorstMetricValue

@dataclass
class Pipeline:
    id: str
    title: str
    description: str 
    code: str
    referenced_private_data: bool
    metric: MetricValue
    submission: Path | None 
    output_dir: Path | None
    suggestions: str | None
    datasets: list[Dataset]

    def __post_init__(self):
        # Assure all referenced files exist
        for dataset in self.datasets:
            if not dataset.base_path.exists():
                raise FileNotFoundError(f"Dataset {dataset} not found")

        # Assure the submission exists if the metric is not WorstMetricValue
        if not isinstance(self.metric, WorstMetricValue) and self.submission is None:
            raise FileNotFoundError(f"Submission {self.output_dir} not found")

    def __str__(self):
        suggestions = f"<suggestions>\n{self.suggestions}\n</suggestions>" if self.suggestions else ""
        datasets = "\n".join(str(dataset) for dataset in self.datasets)
        return f"""
<pipeline>
<title>{self.title}</title>
<id>{self.id}</id>
<referenced_private_data>{self.referenced_private_data}</referenced_private_data>
<description>\n{self.description}\n</description>
<code>\n{self.code}\n</code>
<metric>\n{self.metric}\n</metric>
{suggestions}
<datasets>
{datasets}
</datasets>
</pipeline>
"""
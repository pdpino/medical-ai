from medai.datasets.report_generation.flat import create_flat_dataloader
from medai.datasets.report_generation.hierarchical import create_hierarchical_dataloader

def get_dataloader_constructor(hierarchical):
    return create_hierarchical_dataloader if hierarchical else create_flat_dataloader

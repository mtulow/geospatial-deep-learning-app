#!/usr/bin/env python3
from fastbook import *
from fastai.vision.widgets import *


# 1. Get a sample of the Planet Dataset from Kaggle with Fastai.
path = untar_data(URLs.PLANET_SAMPLE)
df = pd.read_csv(path/"labels.csv")
df['image_name'] = df['image_name']+'.jpg'


# 2. Create a data loader with Fastai.
planet_dls = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                splitter=RandomSplitter(valid_pct=0.2, seed=42),
                get_x=ColReader(0, pref=path/'train'),
                get_y=ColReader(1, label_delim=' '),
                item_tfms=Resize(224),
                batch_tfms=aug_transforms()
            ).dataloaders(df)


# 3. Train model with Resnet50.
learn = vision_learner(planet_dls, resnet50, metrics=accuracy_multi)
learn.fine_tune(2, 3e-2)


# 4. Export the model
learn.path = Path('.')
learn.export()
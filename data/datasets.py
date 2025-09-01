import os
import random
from torch.utils.data import Dataset
from typing import Any, Callable, TypeVar, Generic, Union
from loop_controller import SizedDataset
# import torch
# from torch.utils.data import Dataset, DataLoader

class Matcher:
  match_string: str
  group: Union[str, None]
  label: Union[str, None]

  def __init__(self, match_string, group=None, label=None):
    self.match_string = match_string
    self.group = group
    self.label = label

LabelType = TypeVar('LabelType')
FeatureType = TypeVar('FeatureType')
OutFeatureType = TypeVar('OutFeatureType')
OutLabelType = TypeVar('OutLabelType')

class DatasetGroup(SizedDataset, Generic[LabelType, FeatureType, OutFeatureType, OutLabelType]):

  data: list[tuple[LabelType, FeatureType]]
  feature_transform: Union[Callable[[FeatureType], OutFeatureType], None]
  label_transform: Union[Callable[[LabelType], OutLabelType], None]

  def __init__(self, data, feature_transform=None, label_transform=None):
    self.data = data
    self.feature_transform = feature_transform
    self.label_transform = label_transform

  def print_data(self):
    for data_row in self.data:
      label = self.label_transform(data_row[0]) if self.label_transform else data_row[0]
      feature = self.feature_transform(data_row[1]) if self.feature_transform else data_row[1]
      print((label, feature))

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx: int):
    data_row = self.data[idx]
    label = self.label_transform(data_row[0]) if self.label_transform else data_row[0]
    feature = self.feature_transform(data_row[1]) if self.feature_transform else data_row[1]
    return feature, label

class FileDataset:
  def __init__(self, matchers: list[Matcher]):
    self.data = {}
    self.data_table_format = []

    for matcher in matchers:
      self.match_on(path_elements=matcher.match_string.strip('/').split('/'), current_group=matcher.group, current_label=matcher.label)

  def print_data(self):
    for row in self.data_table_format:
      print(row)

  def add_data_row(self, group, label, feature):
    if group is None:
      raise ValueError('Group must be specified')

    if label is None:
      raise ValueError('Label must be specified')

    if group not in self.data:
      self.data[group] = {}
    if label not in self.data[group]:
      self.data[group][label] = []

    self.data[group][label].append(feature)
    self.data_table_format.append((group, label, feature))

  def enumerate_dir(self, index, path_elements, current_path):
    paths = os.listdir(os.path.join(*current_path))
    remaining_path_elements = path_elements[index + 1:] if len(path_elements) > index else []
    return remaining_path_elements, paths

  def match_on(self, path_elements, current_group=None, current_label=None, current_path=None):
    if current_path is None:
      current_path = ['.']

    # Check if current path is actually a file, if so return
    if os.path.isfile( os.path.join(*current_path) ):
      print('[DataMatcher] Tried to match on file')
      return

    for index, path_item in enumerate(path_elements):
      if path_item == '*':
        remaining_path_elements, paths = self.enumerate_dir(index, path_elements, current_path)
        for path in paths:
          self.match_on(remaining_path_elements, current_group=current_group, current_label=current_label, current_path=current_path + [path])

        break

      elif path_item.startswith('{') and path_item.endswith('}'):
        # Some kind of data element that will set the group/label/be a feature
        if len(path_item) <= 2:
          raise ValueError('Path element could not be parsed')
        
        specifier = path_item[1:-1].split(':')

        if specifier[0] == 'group':
          remaining_path_elements, paths = self.enumerate_dir(index, path_elements, current_path)
          for path in paths:
            self.match_on(remaining_path_elements, current_group=path, current_label=current_label, current_path=current_path + [path])
          break

        if specifier[0] == 'label':
          remaining_path_elements, paths = self.enumerate_dir(index, path_elements, current_path)
          for path in paths:
            self.match_on(remaining_path_elements, current_group=current_group, current_label=path, current_path=current_path + [path])
          break

        if specifier[0] == 'feature':
          # These are data points, let's add them
          for path in os.listdir(os.path.join(*current_path)):
            # Must be a file for us to add it
            if not os.path.isfile(os.path.join(*(current_path + [path]))):
              print('[DataMatcher] Found non-file item in feature directory, skipping')
              continue
            
            # 'feature:full_path' can be specified to add the entire filepath
            feature = os.path.join(*(current_path + [path])) if specifier[1] == 'full_path' else path
            self.add_data_row(current_group, current_label, feature=feature)

      else:
        current_path.append(path_item)
        
  def group(self, group, l_transform=None, f_transform=None, shuffle=True) -> DatasetGroup:
    group_data = self.data[group]

    database = []
    for label in group_data:
      for item in group_data[label]:
        database.append((label, item))
    
    if shuffle:
      random.shuffle(database)
    
    return DatasetGroup(database, feature_transform=f_transform, label_transform=l_transform)



# dataset = FileDataset([
#   Matcher('haha/{group}/*/{label}/{feature:full_path}'), 
#   Matcher('haha/{group}/*/{feature:full_path}', label='bad')
# ])

# train = dataset.group('train', l_transform=lambda x: x.upper())
# test = dataset.group('test')

# print('TRAIN:')
# train.print_data()

# print('\n\nTEST:')
# test.print_data()



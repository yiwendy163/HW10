import pandas as pd
from audio_utils import fft
from data_utils import KMeansClustering, PCA, StandardScaler, SVC

class AwesomeClassifier:
  def __init__(self, pca_components=16):
    self.scaler = StandardScaler()
    self.pca = PCA(n_components=pca_components)
    self.classifier = SVC()
  
  def transform(self, features, train=False):
    raise("Transform must be implemented")

  def fit(self, features, labels):
    features_df = self.transform(features, train=True)
    self.classifier.fit(features_df, labels)

  def predict(self, features):
    features_df = self.transform(features, train=False)
    return self.classifier.predict(features_df)


class AwesomeAudioClassifier(AwesomeClassifier):
  def __init__(self, pca_components=16):
    super().__init__(pca_components)

  def ffts(self, features):
    ffts = []
    for idx, samples in features.iterrows():
      fft_energy, fft_freqs = fft(samples)
      ffts.append(fft_energy)
    return pd.DataFrame(ffts)
  
  def transform(self, features, train=False):
    unscaled_df = self.ffts(features)
    if train:
      scaled_df = self.scaler.fit_transform(unscaled_df)
      features_df = self.pca.fit_transform(scaled_df)
      print("Explained Variance:", self.pca.explained_variance())
    else:
      scaled_df = self.scaler.transform(unscaled_df)
      features_df = self.pca.transform(scaled_df)
    return features_df


class AwesomeImageClassifier(AwesomeClassifier):
  def __init__(self, pca_components=16, num_colors=16):
    super().__init__(pca_components)
    self.clustering = KMeansClustering(n_clusters=num_colors)

  @staticmethod
  def byRGB(C):
    return C["R"] * 256 * 256 + C["G"] * 256 + C["B"]

  def luminances(self, features):
    image_data = []
    for idx, pixels in list(features.iterrows()):
      pixel_luminances = {f"P{i}": (r+g+b)//3 for i,(r,g,b) in enumerate(pixels)}
      image_data.append(pixel_luminances)

    return pd.DataFrame(image_data)

  def palettes(self, features):
    image_data = []
    for idx, pixels in list(features.iterrows()):
      pxs_df = pd.DataFrame(pixels.tolist())
      self.clustering.fit_predict(pxs_df)
      palette = [{"R":int(r), "G":int(g), "B":int(b)} for r,g,b in self.clustering.cluster_centers_]
      palette_sorted = sorted(palette, key=AwesomeImageClassifier.byRGB)
      palette_obj = {f"{ch}{co}": v for co,pixel in enumerate(palette_sorted) for ch, v in pixel.items()}
      image_data.append(palette_obj)

    return pd.DataFrame(image_data)

  def transform(self, features, train=False):
    palettes_df = self.palettes(features)
    unscaled_df = self.luminances(features)
    if train:
      scaled_df = self.scaler.fit_transform(unscaled_df)
      pca_df = self.pca.fit_transform(scaled_df)
      print("Explained Variance:", self.pca.explained_variance())
    else:
      scaled_df = self.scaler.transform(unscaled_df)
      pca_df = self.pca.transform(scaled_df)

    return pd.concat([pca_df, palettes_df], axis=1)

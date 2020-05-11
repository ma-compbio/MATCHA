import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


vec = np.load("../embeddings.npy")
label = np.load("../subcompartment_label_hg38_1Mb.npy")
vec = vec[label != -1]
label = label[label != -1]
vec = PCA(n_components=2).fit_transform(vec)



label = np.array(["State"+str(label[i]) for i in range(len(label))])
g = sns.scatterplot(x = vec[:,0],y = vec[:,1],hue = label,alpha = 1.0,linewidth=0, s = 30, )
plt.savefig("../scatter.png")
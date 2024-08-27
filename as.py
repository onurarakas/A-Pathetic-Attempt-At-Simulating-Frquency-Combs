import matplotlib.pylab as plt
import matplotlib.transforms





f,a = plt.subplots(figsize=(8.4,12))

boxdata = [[4,4,5,8],[4,4,5,8],[4,4,5,8]]
box = a.boxplot(boxdata)
labels = ["aaaaaaa","AAAAAAA","iiiiiii"]

# Create an offset in the direction Y
dx = 0/72.; dy = -50/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in a.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

# apply a rotation and align in the bottom
a.set_xticklabels(labels, rotation=90, verticalalignment="bottom")
plt.show()

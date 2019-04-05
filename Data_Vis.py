import matplotlib.pyplot as plt
import seaborn as sns
#Generates a scatter plot by geographic location of training data.
#Categories or values of var determine the color
#Skips any training values with missing location data

var='lga'
sns.scatterplot(x='longitude', y='latitude',
                data=TrnVal[TrnVal.longitude !=0],hue=var)
plt.show()

import matplotlib.pyplot as plt
 
labels = ['Wake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
sizes = [25.63, 6.59, 39.56, 16.30, 11.92]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'green']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.savefig('class_dist.jpg')

import pandas as pd
df = pd.read_csv('/Users/mashaojie/PyCharmMiscProject/Dataset2_theme.csv', encoding='utf-8-sig')
#select the data range
df = df[['reviews.rating','reviews.title','reviews.text', 'theme']]
# create the keywords column
df['keywords'] = ''
#show the data theme 
print(df['theme'].unique())
print(f"theme_amount:{len(df['theme'].unique())}")
#import the dataset1
df1 = pd.read_csv("/Users/mashaojie/Desktop/Neoma/S4/毕业论文/Database/Dataset1.csv", encoding='utf-8-sig')

# modify the theme and fill the keywords
# Cleaniess
mask = df['theme'].isin(['1', '-1_0', '-1_4', '-1_clean'])
df.loc[mask, 'theme'] = 'Cleanliness'
df.loc[mask, 'keywords'] = 'clean, bed, bathroom, shower, room, fridge, microwave'

# Location
mask = df['theme'].isin(['3', '-1_1', '-1_2', '-1_3', '-1_location'])
df.loc[mask, 'theme'] = 'Location'
df.loc[mask, 'keywords'] = 'hotel, nearby, distance, center, transport, accessibility, convenience'

# Service
mask = df['theme'].isin(['0', '-1_0', '-1_3', '-1_staff'])
df.loc[mask, 'theme'] = 'Service'
df.loc[mask, 'keywords'] = 'staff, front, service, desk, time'

# Food & Drinks
mask = df['theme'].isin(['2', '-1_breakfast'])
df.loc[mask, 'theme'] = 'Food & Drinks'
df.loc[mask, 'keywords'] = 'breakfast, restaurant, lunch , dinner'

# Else
mask = df['theme'].isin(['4', '-1_4'])
df.loc[mask, 'theme'] = 'Else'
df.loc[mask, 'keywords'] = 'noise, night, air'

# Else
mask = df['theme'].isin(['-1_else'])
df.loc[mask, 'theme'] = '0'
df.loc[mask, 'keywords'] = '0'

#save the Dataset1_theme_keywords.csv
df1['theme'] = df['theme']
df1['keywords'] = df['keywords']
df1 = df1[df1['theme'] != '0']
df1.to_csv('Dataset1_theme_keywords.csv', index=False, encoding='utf-8-sig')

#save the results in Dataset2_theme_keywords.csv
df = df[df['theme'] != '0']
df.to_csv('Dataset2_theme_keywords.csv', index=False, encoding='utf-8-sig')
print(df['theme'].value_counts())





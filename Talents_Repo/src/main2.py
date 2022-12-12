import main

df, search_terms, similarity_index = main.main_func("data\\raw\\potential-talents - Aspiring human resources - seeking human resources.csv")

#set some ideal candidates
df.at[4, 'fit'] = 1
df.at[5, 'fit'] = 1
df.at[6, 'fit'] = 1
df.at[7, 'fit'] = 1

df_rough_final = main.main_func2(df, search_terms, similarity_index)

df_final = main.clean_and_save(df_rough_final, "data\\processed\\sorted_list.csv")

print(df_final)
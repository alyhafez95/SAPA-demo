# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# all depmap samples' metadata
sample_metadata = spark.read.csv('/mnt/dev-zone0/csv/rdm/depmap/sample_info', header=True)
display(sample_metadata)

# COMMAND ----------

import plotly.express as px
x=sample_metadata.groupby('primary_disease').count().sort(F.col("count").desc()).toPandas()
fig = px.bar(x, x='primary_disease', y='count')
fig.show()

# COMMAND ----------

x=sample_metadata.groupby('sample_collection_site').count().sort(F.col("count").desc()).toPandas()
fig = px.bar(x, x='sample_collection_site', y='count')
fig.show()

# COMMAND ----------

expression_levels = spark.read.load('/mnt/dev-zone1/parquet/rdm/depmap/expression_levels', format='parquet')
display(expression_levels)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Our goal here is to create a more usable gene expression table with sample information and proper gene ids 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Step One Join Samples table and gene expression table to get the gene expression profile of each sample

# COMMAND ----------

samples_expression_levels = sample_metadata.join(expression_levels, (sample_metadata['DepMap_ID'] == expression_levels['depmap_id']), how='left') \
    .drop(expression_levels['depmap_id'])

# COMMAND ----------

print(samples_expression_levels.count())
samples_expression_levels.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC that was pretty fast for a join oberation involving 2.6 million rows, lets see how pandas handles this

# COMMAND ----------

sample_metadata_pd = sample_metadata.toPandas()
sample_metadata_pd['depmap_id'] = sample_metadata_pd['DepMap_ID'].astype(str)
expression_levels_pd = expression_levels.toPandas()

# COMMAND ----------

samples_expression_levels_pd = sample_metadata_pd.join(expression_levels_pd.set_index('depmap_id'), on='depmap_id', how='left')

# COMMAND ----------

samples_expression_levels_pd = samples_expression_levels_pd.join(samples_expression_levels_pd.set_index('depmap_id'), on='depmap_id', how='inner')


# COMMAND ----------

samples_expression_levels_test = samples_expression_levels.join(samples_expression_levels, ['depmap_id'], how='inner') \
    .drop(expression_levels['depmap_id'])
print(samples_expression_levels_test.count())
samples_expression_levels_test.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC lets see what the expression profile of a single sample looks like. We will use sample ACH-000001

# COMMAND ----------

x= samples_expression_levels.where('DepMap_ID == "ACH-000001"') \
    .select(['gene_symbol', 'expression_level']) \
    .sort(F.col("expression_level").desc()).toPandas()
    
print(len(x))
fig = px.bar(x, x='gene_symbol', y='expression_level')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC very nice, we can see that at least this specific sample has a lot of mitochondrial gene expression; must be very active!

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Now we have to resolve an issue regarding the unreliablity of using gene symbols as a unique identifer for genes. The issue is gene symbols are great when working within the same org or research team when the research context is well understood and sample type, species, refernce are all known. but you find that they are highly unrealiable when trying to use them universily outside of a signle team/org. you can you read more about this issue here:
# MAGIC https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1480234/

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC * For this we are going to use a databse of controlled gene symbols and the universal IDs that they map to (ensembl IDs) additinally we are going to be on the lookout for gene synonyms which is the case when more than one gene symbol map to a single gene ensembl id.
# MAGIC

# COMMAND ----------

# load genes table
gene_data = spark.read.load('/mnt/dev-zone2/parquet/rdm/database/rdm_metadata_dev/dim_gene', format='parquet')
print(gene_data.count())
gene_data.display()

# COMMAND ----------

# load table of genes synonyms 
gene_synonyms = spark.read.load('/mnt/dev-zone2/parquet/rdm/database/rdm_metadata_dev/dim_gene_synonym', format='parquet')
print(gene_synonyms.count())
gene_synonyms.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC * there is a lot of curation involved when picking up the genes that you want, you have to make sure you're picking up only the relevant genes
# MAGIC * I built this function to help me do that

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

def gene_genie(species_list = [],
               include_synonyms = True,
               native_chromosomes_only = True,
               exclude_lrg = True,
               case_insentive = True,
               **kwargs):
    '''
    This tool is a multipurpose gene utility with opitons to
        - To compine dim_gene and dim_gene_synonym 
        - Filter gene data from irrelevant species 
        - Filter gene data from mutated or experimental chromosomes
       

    Attributes:
        species_list: list of species to include 
        include_synonyms: option to include gene symbol synonyms in resulting data as additional rows
        native_chromosomes_only: option to filter out gene data from mutated or experimental chromosomes
        exclude_lrg: option to filter out gene data that is known as Locus Reference Genomic or LRG (used for reporting sequence variants with clinical implications)

        **kwargs: ad-hoc keyword arguments as column_name = ['list of values']
          Ex.
            gene_name = ['C-C motif chemokine receptor 3']
            gene_type = ['protein_coding']

    Returns:
        filtered_gene_df: a df with relevant gene data as specified by the user
    '''
    try:
        #  dim_gene & dim_gene_synonym from rdm database
        dim_gene = spark.read.load('/mnt/dev-zone2/delta/rdm/database/metadata/dim_gene', format = 'delta')
        dim_gene_synonym = spark.read.load('/mnt/dev-zone2/delta/rdm/database/metadata/dim_gene_synonym', format = 'delta')

        # dict to map species with their common name 
        species_dict = {'human': 'homo_sapiens_core_100_38',
                        'mouse': 'mus_musculus_core_100_38',
                        'rat': 'rattus_norvegicus_core_100_6',
                        'green_monkey': 'chlorocebus_sabaeus_core_100_1',
                        'cynomolgus_monkey': 'macaca_fascicularis_core_103_6'}

        ref_genome = [species_dict[species] for species in species_list]

        if len(species_list) != 0:
            dim_gene = dim_gene.filter(F.col('species_name').isin(ref_genome))

        # filter out non native chromosomes
        if native_chromosomes_only:
            dim_gene = dim_gene.filter(dim_gene.seq_name.rlike("^[1-9]{1,2}") |
                                       dim_gene.seq_name.rlike("(?i)^[XYxy]$") |
                                       dim_gene.seq_name.rlike("(?i)^mt$"))
        # filter out LRG genes
        if exclude_lrg:
            dim_gene = dim_gene.filter(dim_gene.gene_id.rlike("^((?!LRG).)*$"))

        # gene symbol description 'alias_type' column and 'primary_gene_symbol' for mapping functionality
        # all gene_symbols in dim_gene are considered primary_gene_symbols
        # gene_symbols that dont duplicate (one of a kind); those will get primary_gene_ids
        ooak_gene_symbols = dim_gene.groupby('gene_symbol','species_name').count().where('count == 1')
        dim_gene = dim_gene.join(ooak_gene_symbols, ['gene_symbol','species_name'], 'left') \
                           .withColumn('alias_type', F.lit('primary')) \
                           .withColumn('primary_gene_symbol', F.lit(F.col('gene_symbol'))) \
                           .withColumn('primary_gene_id', F.when(F.col('count') == 1, F.col('gene_id')).otherwise(F.lit(None)) ) \
                           .drop('count')


        if include_synonyms:
          # rename all the columns in the dim_gene_synonym table so it is easier to union with dim_gene
            for column in dim_gene_synonym.columns:
                  dim_gene_synonym = dim_gene_synonym.withColumnRenamed(column, f'syn_{column}')
            # join dim_gene_synonym with dim_gene (to add gene info to synonyms) 
            gene_synonym_df = dim_gene_synonym.join(dim_gene, (dim_gene_synonym['syn_Gene_Id'] == dim_gene['gene_id']), how= 'inner').drop(dim_gene['gene_symbol']) \
                                              .withColumn('alias_type', F.lit('synonym')) \
                                              .withColumnRenamed('syn_synonym', 'gene_symbol') \
                                              .select(*dim_gene.columns)

            # vertically union dim_gene_synonym with dim_gene to have all gene_symbols + synonyms in the same column
            dim_gene = dim_gene.union(gene_synonym_df)
            
        if case_insentive:
            dim_gene = dim_gene.withColumn('gene_symbol', F.lower(F.col('gene_symbol')))
            
        # ad hoc filters functionality 
        for key, value in kwargs.items():
            dim_gene = dim_gene.filter(F.col(f'{key}').isin(value))
        
    except Exception as e:
        return log_exception(prod = False, source = 'dim_gene', process_id = None, batch_id = None, workflow = 'gene_annotation',
                              function = "gene_genie()", exception = e)
      
    return dim_gene

#Note:

# the resulting dataframe may contain multiple synonyms per gene_sympol
# and/or duplicate gene_sympol if multiple species are selected
# and/or duplicate gene_sympol if symbol is shared by two genes via synonyms
#            meaning if the gene_sympol is used by two different genes but one of the genes has a primary_gene_sympol = another's gene's synonyms
# there may be multiple primary gene_id's for the same gene within a species as the same gene show up in different loci.
#            ex. SNORA75 (gene_symbol) shows up 7 times in the human genes list (all listed as primary) due to the gene symbol showing up in 7 different locations across different chromosomes.
# there maybe null primary gene_ids if there is duplicate matches to that gene symbol with differening gene_ids (can't arbitrarily pick one so we pick null)

# Example use: 
# gene_df = gene_genie(species_list = ['human', 'mouse', 'rat', 'green_monkey', 'cynomolgus_monkey'],
#                                      include_synonyms = True,
#                                      native_chromosomes_only = True,
#                                      exclude_lrg = True)


# COMMAND ----------

all_human_genes_df = gene_genie(species_list = ['human'],
                                     include_synonyms = True,
                                     native_chromosomes_only = True,
                                     exclude_lrg = True)
all_human_genes_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC now we have a dataset of only the native human genes, along with their synonyms

# COMMAND ----------

samples_expression_levels_fact = (samples_expression_levels.alias("se").join(all_human_genes_df.dropDuplicates(['ncbi_id']), 
                                                        (expression_levels['entrez_id'] == all_human_genes_df['ncbi_id']), 
                                                        how= 'left'))

unmatched_rows = samples_expression_levels_fact.where("primary_gene_id is null").select(["se.*"])

samples_expression_levels_fact = samples_expression_levels_fact.where("primary_gene_id is not null")


print('number of matched rows:', samples_expression_levels_fact.count())
print('number of unmatched rows:', unmatched_rows.count())
samples_expression_levels_fact.display()

# COMMAND ----------

unmatched_rows = unmatched_rows.withColumn('gene_symbol', F.lower(F.col('gene_symbol')))
samples_expression_levels_fact2 = (unmatched_rows.join(all_human_genes_df.dropDuplicates(['gene_symbol']), 
                                                        (unmatched_rows['gene_symbol'] == all_human_genes_df['gene_symbol']), 
                                                        how= 'left'))
print(samples_expression_levels_fact2.count())
samples_expression_levels_fact2.display()

# COMMAND ----------

samples_expression_levels_fact_final = samples_expression_levels_fact.union(samples_expression_levels_fact2)
print(samples_expression_levels_fact_final.count())
samples_expression_levels_fact_final.display()

# COMMAND ----------

samples_expression_levels_fact_final.where('primary_gene_id is null').count()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC With that we have fully annotated the gene expression dataset with the gene symbols that we internally use and the ensembl IDs that are widely used/accepted

# COMMAND ----------


x= samples_expression_levels_fact_final.where('DepMap_ID == "ACH-000001"') \
    .select(['primary_gene_symbol', 'expression_level']) \
    .sort(F.col("expression_level").desc()).toPandas()
    
print(len(x))
fig = px.bar(x, x='primary_gene_symbol', y='expression_level')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC and with that we're pretty much done with this stage of data processing; lets go ahead and save our data

# COMMAND ----------

# samples_expression_levels_fact_final.write.saveAsTable('DepMap_samples_expression_levels',
#                               format = 'delta',
#                               mode = 'append',
#                               path = '/mnt/dev-zone2/delta/warehouse/DepMap_samples_expression_levels')

# COMMAND ----------



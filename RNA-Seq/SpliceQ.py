''' RUN IN UBUNTU
import pysam
import pandas as pd
# The splicing efficiency can be measured in different ways, such us: 
file =  "RNA-Seq/WT41.Aligned.sortedByCoord.out.bam"


read_id = []
chromosome = []
position = []
map_qual = []
read_seq = []
 
# 1. BAM file
with pysam.AlignmentFile(file, "rb") as bamfile:
    head = bamfile.header
    print(head)
    for alignment in bamfile:
        read_id.append(alignment.query_name)
        chromosome.append(alignment.reference_name)
        position.append(alignment.reference_start)
        map_qual.append(alignment.mapping_quality)
        read_seq.append(alignment.query_sequence)

# 2. Dataframe creation
dictionary = {"read": read_id, "reference_name": chromosome, "position": position, "mapping_quality": map_qual, "sequence": read_seq}
rnaseq_info = pd.DataFrame(dictionary)

# print(rnaseq_info)
'''






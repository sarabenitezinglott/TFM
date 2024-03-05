import bamnostic as bs
file = "D:\\TFM\\RNAseq\\BAM files\\WT41.Aligned.sortedByCoord.out.bam" 
bam = bs.AlignmentFile(file, "rb")
# 1. Get the attributes of bam 
print(dir(bam))
# 2. The data at the heather of bam
print(bam.header)
# 3. Next aligment
align = next(bam)
print(align)
# 4. Loop inside bam to get all the information
for alignment in bam:
    print(alignment.reference_name, alignment.pos)





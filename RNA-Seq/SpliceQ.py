import bamnostic as bs
file = "D:\\TFM\\RNAseq\\BAM files\\WT41.Aligned.sortedByCoord.out.bam" 
bam = bs.AlignmentFile(file, "rb")
print(bam.header)




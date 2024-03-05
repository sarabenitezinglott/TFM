import pysam

file =  "RNA-Seq/WT41.Aligned.sortedByCoord.out.bam"
bamfile = pysam.AlignmentFile(file, "rb")
print(bamfile.count)







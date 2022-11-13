import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob

def finalBarplot(geneListPath, KOfile, OEfile, sign=False):
    colors = {"KO" : "#ba7ae9", "OE" : "#ff0049"}
    genes = pd.read_csv(geneListPath, delimiter="\t")
    genes.columns = ["Name", "Id"]
    if not sign:
        ko = pd.read_csv(KOfile, delimiter = "\t", low_memory = False)
        oe = pd.read_csv(OEfile, delimiter = "\t", low_memory = False)
        ko = ko.merge(genes, on = ["Id"])
        oe = oe.merge(genes, on = ["Id"])
        ko["Condition"] = ["KO"] * ko.shape[0]
        ko["Colors"] = colors["KO"] * ko.shape[0]
        oe["Condition"] = ["OE"] * oe.shape[0]
        oe["Colors"] = colors["OE"] * ko.shape[0]
        final = pd.concat([ko, oe])
        if len(set(final["Name"])) == final.shape[0] / 2:
            ax = sns.barplot(data = final, y = "Name", x = "log2FoldChange", hue = "Condition", palette = colors)
        else:
            ax = sns.barplot(data = final, y = "Id", x = "log2FoldChange", hue = "Condition", palette = colors)
        ax.set_xlabel("Log2 Fold Change",fontsize=24)
        ax.set_ylabel("")
        plt.tick_params(labelsize=22)
        plt.yticks(np.arange(0, genes.shape[0], 1))
        plt.axvline(x=0, color="black", linewidth=1, linestyle="--")
        fig = plt.gcf()
        fig.set_size_inches(46, 9)
        fig.savefig(geneListPath.replace(".txt", ".png"))
        plt.clf()
    else:
        df = pd.read_csv(KOfile, delimiter="\t", low_memory=False, index_col = 0)
        df["Id"] = df.index.tolist()
        df = df.merge(genes, on = ["Id"])
        ko = df[~pd.isna(df["log2FoldChange_KOvsWT"])]
        oe = df[~pd.isna(df["log2FoldChange_KOvsWT"])]
        if ko.shape[0] != 0:
            ko = ko[["Name", "log2FoldChange_KOvsWT", "Id", "Cluster"]]
            ko.columns = ["Name", "log2FoldChange", "Id", "Cluster"]
            ko["Condition"] = ["KO"] * ko.shape[0]
        if ko.shape[0] != 0:
            oe = oe[["Name", "log2FoldChange_OEvsWT", "Id", "Cluster"]]
            oe.columns = ["Name", "log2FoldChange", "Id", "Cluster"]
            oe["Condition"] = ["OE"] * oe.shape[0]
        final = pd.concat([ko, oe])
        if final.shape[0] > 0:
            final = final[~pd.isna(final["log2FoldChange"])]
            if final.shape[0] > 0:
                final.to_excel(geneListPath.replace(".txt", "_sign.xlsx"))
                ax = sns.barplot(data = final, y = "Name", x = "log2FoldChange", hue = "Condition", palette = colors)
                ax.set_xlabel("Log2 Fold Change",fontsize=24)
                ax.set_ylabel("")
                plt.tick_params(labelsize=22)
                plt.yticks(np.arange(0, genes.shape[0], 1))
                plt.axvline(x=0, color="black", linewidth=1, linestyle="--")
                fig = plt.gcf()
                fig.set_size_inches(46, 9)
                fig.savefig(geneListPath.replace(".txt", "_sign.png"))
                plt.clf()

if __name__ == "__main__":
    for f in glob.glob("/home/kevin/Bureau/RNASeq_Scripts/RNASeq/Barplots/*.txt"):
        print(f)
        finalBarplot(f, "/home/kevin/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/KOvsWT/KOvsWT.complete.txt", "/home/kevin/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/OEvsWT/OEvsWT.complete.txt", "")
        finalBarplot(f, "/home/kevin/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign/K_7/k_7.tsv", "", True)
    
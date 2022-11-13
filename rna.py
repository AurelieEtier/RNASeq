import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib
import subprocess
from math import sqrt
import re
from collections import defaultdict
import pylab

def computeMean(row):
    return row.mean()

def computeStd(row):
    return row.std()

def stoufferMethod(zscoresReplicate):
    zscoresFinal = []
    for i in range(0,len(zscoresReplicate), 3):
        zscoresFinal.append(str(sum(zscoresReplicate[i:i+3]) / sqrt(len(zscoresReplicate))))
    return " ".join(zscoresFinal)

def computeZscore(row, columns):
    zscoresReplicate = []
    for c in columns:
        zscoresReplicate.append((row[c] - row["Mean"]) / row["Std"])
    return stoufferMethod(zscoresReplicate)

def zScore(df):
    new_columns = {"Z score WT" : "Z_WT", "Z score OE" : "Z_OE", "Z score KO" : "Z_KO"}
    columns = []
    for col in df.columns.tolist():
        if "norm." in col:
            columns.append(col)
    columns = [value for value in columns if not "cH2AZ" in value]
    df["Mean"] = df[columns].apply(lambda row_: computeMean(row_), axis=1)
    df["Std"] = df[columns].apply(lambda row_: computeStd(row_), axis=1)
    df = df[df["Std"] != 0]
    if df.shape[0] == 0:
        return pd.DataFrame()
    df["Z score"] = df.apply(lambda row_: computeZscore(row_, columns), axis=1)
    df[['Z score WT','Z score KO', "Z score OE"]] = df["Z score"].str.split(" ",expand=True,)
    df['Z score WT'] = pd.to_numeric(df['Z score WT'])
    df['Z score KO'] = pd.to_numeric(df['Z score KO'])
    df['Z score OE'] = pd.to_numeric(df['Z score OE'])
    df.pop("Mean")
    df.pop("Std")
    df.pop("Z score")
    df = df.rename(columns = new_columns)
    return df

def pca(df, outDir, removeSamples):
    columns = df.columns.tolist()
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    toSort = dict(zip(columns, order))
    if len(removeSamples) > 0:
        a = pd.DataFrame()
        columns = list(set(columns).difference(set(removeSamples)))
        a["Col"] = columns
        a["Sort"] = [toSort[col] for col in columns]
        a = a.sort_values(by=["Sort"])
        columns = a["Col"].tolist()

    df = df[columns]
    df.index = df["Id"].tolist()
    df.pop("Id")
    expressed = []
    for index, row in df.iterrows():
        rowToList = row.tolist()
        rowToSet = set(rowToList)
        if len(rowToSet) == 1:
            for value in rowToSet:
                if value == 0:
                    expressed.append(False)
                else:
                    expressed.append(True)
        else:
            expressed.append(True)
    df["Expressed"] = expressed
    df = df[df["Expressed"] == True]
    df.pop("Expressed")
    df = df.T
    return df

def scatterPlot(table, outDir):
    matplotlib.rc('font',family='Courier')
    listConditions = ["dH2AZ", "WT", "OEH2AZ"]
    df = pd.read_csv(table , sep = "\t")
    df = df[["Z_WT", "Z_KO", "Z_OE", "Cluster"]]
    df.columns = ["WT", "dH2AZ", "OEH2AZ", "Cluster"]
    nbYPlots = 4
    nbXPlots = len(list(set(df["Cluster"]))) / nbYPlots if len(list(set(df["Cluster"]))) % nbYPlots == 0 else len(set(df["Cluster"])) / nbYPlots + 1
    y = []
    x = []
    plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=0.5, wspace = 0.8)
    plt.suptitle(f"Kmeans (K = {len(set(df['Cluster']))})", fontsize=18, y=1.5)
    clusters = list(set(df["Cluster"]))
    for i in range(len(clusters)):
            scatterDf = pd.DataFrame()
            ax = plt.subplot(int(nbXPlots), nbYPlots, i+1)
            df2 = df[df["Cluster"] == clusters[i]]
            y = []
            x = []
            for cond in listConditions:
                columns = [value for value in df2.columns.tolist() if cond in value]
                subDf = df2[columns]
                for col in subDf.columns.tolist():
                    y += subDf[col].tolist()
                    x += [cond] * len(subDf[col].tolist())
            scatterDf["Z score"] = y
            scatterDf["Conditions"] = x
            meanDf = computeMeanPointScatterPlot(scatterDf, listConditions)
            scatterDf = pd.concat([meanDf, scatterDf])
            scatterDf["Type"] = scatterDf["Type"].fillna("")
            s = sns.scatterplot(data=scatterDf, x = "Conditions", y = "Z score", hue = "Type")
            s.tick_params(axis='x', which='major', labelsize=18)
            s.tick_params(axis='y', which='major', labelsize=18)
            conditions = scatterDf[scatterDf["Type"] == "Mean"].sort_values(by = ["Conditions"])["Conditions"].tolist()
            values = scatterDf[scatterDf["Type"] == "Mean"].sort_values(by = ["Conditions"])["Z score"].tolist()
            x = np.array([conditions[0], conditions[1], conditions[2]])
            y = np.array([values[0], values[1], values[2]])
            plt.plot(x, y)
            plt.title(f" {clusters[i]} (n = {df2.shape[0]})", fontsize = 18)
            plt.xlabel('', fontsize=18)
            if i % nbYPlots != 0:
                plt.ylabel('', fontsize=18)
            else:
                plt.ylabel('Z score', fontsize=18)
            pylab.rcParams['xtick.major.pad']='12'
            ax.get_legend().remove()
            plt.ylim(-1.5, 1.5)
            plt.savefig(f"{outDir}/profile_{len(set(df['Cluster']))}.png")

def computeMeanPointScatterPlot(scatterDf, listConditions):
    meanDf = pd.DataFrame()
    means = []
    conditions = []
    for condition in listConditions:
        means.append(scatterDf[scatterDf["Conditions"] == condition]["Z score"].median())
        conditions.append(condition)
    meanDf["Z score"] = means
    meanDf["Conditions"] = conditions
    meanDf["Type"] = ["Mean"] * meanDf.shape[0]
    return meanDf

def clusterColors(df):
    df = df.sort_values(by = "Cluster")
    clusters = df.pop("Cluster")
    clustersNb = set(clusters)
    cmap1 = matplotlib.cm.get_cmap('tab20c')
    cmap2 = matplotlib.cm.get_cmap('tab20b')
    tab20c = [matplotlib.colors.rgb2hex(cmap1(i)[:3]) for i in range(cmap1.N)]
    tab20b = [matplotlib.colors.rgb2hex(cmap1(i)[:3]) for i in range(cmap1.N)]
    tab = tab20c + tab20b
    colors = dict(zip(clustersNb, tab))
    return colors

def clusterMap(df, outDir, colors, mode):
    df = df.sort_values(by = ["Cluster", "Z_WT"])
    clusters = df.pop("Cluster")
    df2 = df[["Z_WT", "Z_KO", "Z_OE"]]
    df2.columns = ["WT", "dH2AZ", "OEH2AZ"]
    row_colors = [colors[value] for value in clusters]
    df2["Cluster"] = row_colors
    row_colors = df2.pop("Cluster")
    c = sns.clustermap(df2, row_colors = row_colors, row_cluster=False, col_cluster=True,  yticklabels=False, cmap = "bwr")
    if mode == "total":
        plt.title(f"K = {len(set(clusters))}")
        plt.savefig(f"{outDir}/heatmap_{len(set(clusters))}.png")
    elif mode == "cluster":
        plt.title(f"Cluster {list(set(clusters))[0]}")
        plt.savefig(f"{outDir}/cluster_{clusters[0]}.png")
    else:
        mode = mode.replace("/", "_")
        plt.title(f"{df.shape[0]} Genes")
        plt.savefig(f"{outDir}/{'_'.join(mode.split(' '))}.png")
        df["Cluster"] = clusters
        df.to_csv(f"{outDir}/{'_'.join(mode.split(' '))}.tsv", sep="\t")


def clusterMapGO(df, outDir, colors, name):
    df = df.sort_values(by = ["Cluster", "Z_WT"])
    clusters = df.pop("Cluster")
    df2 = df[["Z_WT", "Z_KO", "Z_OE"]]
    df2.columns = ["WT", "dH2AZ", "OEH2AZ"]
    row_colors = [colors[value] for value in clusters]
    df2["Cluster"] = row_colors
    row_colors = df2.pop("Cluster")
    c = sns.clustermap(df2, row_colors = row_colors, row_cluster=False, col_cluster=True,  yticklabels=False, cmap = "bwr")
    plt.title(f"{name}")
    plt.savefig(f"{outDir}/{name}.png")
    df.to_excel(f"{outDir}/{name}.xlsx")

def filterVariant(pcaTable, theshold, outDir):
    df = pd.read_csv(pcaTable, sep="\t", index_col=0)
    df = df.T
    df["Variance"] = df.var(axis=1)
    df = df[df["Variance"] >= theshold]
    df.pop("Variance")
    df = df.T
    df.to_csv(f"{outDir}/KOvsWT.pca_filtre.txt", sep="\t")

def withVariance(inFile, outDir, threshold):
    outDir = f"{outDir}/Variance_{threshold}"
    subprocess.call(f"mkdir -p {outDir}", shell=True)
    df = filterVariant(inFile, threshold, outDir)
    subprocess.call(f"Rscript /home/aurelie/Bureau/RNASeq_Scripts/RNASeq/pca.R {outDir}/KOvsWT.pca_filtre.txt {outDir}/", shell=True)
    files = glob.glob(f"{outDir}/*tsv")
    for f in files:
        df = pd.read_csv(f, sep="\t")
        clusterMap(df, outDir)

def withConcat(inDir, outDir, removeSamples, columnsToKeep):
    subprocess.call(f"mkdir -p {outDir}", shell=True)
    for i in range(2, 31):
        subprocess.call(f"mkdir -p {outDir}/K_{i}/", shell=True)
    conditions = ["KOvsWT", "OEvsKO", "OEvsWT"]
    complete = pd.DataFrame()
    for condition in conditions:
        df = pd.read_csv(f"{inDir}/{condition}/{condition}.complete.txt", sep="\t")
        df["Condition"] = [condition] * df.shape[0]
        complete = pd.concat([complete, df])
    df = concatConditions(complete)
    df = pca(df, outDir, removeSamples)
    df = zScore(df.T)
    df = df.sort_values(by = ["Z_WT"])
    df["Z_WT"] = df["Z_WT"].astype(float)
    df["Z_KO"] = df["Z_KO"].astype(float)
    df["Z_OE"] = df["Z_OE"].astype(float)
    df = df.T
    df.to_csv(f"{outDir}/pca.txt", sep="\t", index=True)
    subprocess.call(f"Rscript /home/aurelie/Bureau/RNASeq_Scripts/RNASeq/pca.R {outDir}/pca.txt {outDir}/ {9 - len(removeSamples)}", shell=True)
    files = glob.glob(f"{outDir}/K_*/*tsv")
    for f in files:
        df = pd.read_csv(f, sep="\t")
        colors = clusterColors(df)
        df2 = pd.read_csv(f, sep="\t")
        outK = os.path.dirname(f)
        clusterMap(df, outDir, colors, "total")
        for cluster in set(df2["Cluster"]):
            df3 = df2[df2["Cluster"] == cluster]
            clusterMap(df3, outK, colors, "cluster")

def addGo(inDir, goFile):
    go = pd.read_csv(goFile, sep="\t")
    files = glob.glob(f"{inDir}/K_*/k_*.tsv")
    for f in files:
        df = pd.read_csv(f, sep = "\t")
        df["Id"] = df.index.tolist()
        df = df.merge(go, on = ["Id"])
        df.index = df["Id"].tolist()
        df.pop("Id")
        df.to_csv(f, sep = "\t")

def main(inDir, outDir, goFile, removeSamples, columnsToKeep):
    withConcat(inDir, outDir, removeSamples, columnsToKeep)
    addGo(outDir, goFile)
    for f in glob.glob(f"{outDir}/K_*/k_*.tsv"):
        scatterPlot(f, outDir)

def searchInGo(pathK, column, outDir):
    subprocess.call(f"mkdir -p {outDir}/{column}", shell=True)
    df = pd.read_csv(pathK, sep="\t", index_col=0)
    a = pd.DataFrame()
    a[column] = list(set((",".join(list(set(df[~pd.isna(df[column])][column]))).split(","))))
    a = a.sort_values(by = [column])
    a.to_csv(f"{outDir}/{column}/{column}.tsv", sep="\t", index=None)
    heatmapKeywords(df, a, outDir, column)

def heatmapKeywords(df, table, outDir, column):
    colors = clusterColors(df)
    column = table.columns.tolist()[0]
    df = df[~pd.isna(df[column])]
    keywords = table[column].tolist() #["chromatin", "aurofusarin", "histone", "metabolite"]
    for keyword in keywords:
            clusterMap(df[df[column].str.contains(re.escape(keyword))], f"{outDir}/{column}", colors, keyword)

def keepColumns(df, columns, condition):
    columnsToKeep = [col for col in df.columns.tolist() if col in columns]
    columns = [f"{col} ({condition})" for col in columns]
    toKeep = dict(zip(columnsToKeep, columns))
    df["log2"] = df["log2FoldChange"]
    df["padj_keep"] = df["padj"]
    df = df.rename(columns = toKeep)
    return df

def concatConditions(df):
    columns = ['Id', 'norm.WT1', 'norm.WT2', 'norm.WT3', 'norm.dH2AZ_1', 'norm.dH2AZ_2', 'norm.dH2AZ_3', 'norm.OEH2AZ_1', 'norm.OEH2AZ_2', 'norm.OEH2AZ_3', 'WT', 'KO', 'OE', 'padj', 'log2FoldChange', 'Condition']
    columnsToKeep = ['Id', 'norm.WT1', 'norm.WT2', 'norm.WT3', 'norm.dH2AZ_1', 'norm.dH2AZ_2', 'norm.dH2AZ_3', 'norm.OEH2AZ_1', 'norm.OEH2AZ_2', 'norm.OEH2AZ_3', 'WT', 'KO', 'OE']
    df = df[columns]
    padj = df[df["padj"] < 0.05]
    newDf = pd.DataFrame()
    for id in set(padj["Id"]):
        subDf = padj[padj["Id"] == id]
        line = defaultdict(list)
        for column in columnsToKeep:
            line[column].append(subDf[column].tolist()[0])
        for index, row in subDf.iterrows():
            line[f"padj_{row['Condition']}"].append(row["padj"])
            line[f"log2FoldChange_{row['Condition']}"].append(row["log2FoldChange"])
        line["Condition"].append("/".join(subDf["Condition"].tolist()))
        newDf = pd.concat([newDf, pd.DataFrame(line)])
    finalColumns = newDf.columns.tolist()
    finalColumns.pop(finalColumns.index("Condition"))
    finalColumns.append("Condition")
    newDf = newDf[finalColumns]
    return newDf

def heatMapByKeywordAll(inDir, k, column, keywordTxt, mode):
    keywords = "".join(open(keywordTxt, "r").readlines()).split("\n")
    outDir = f"{inDir}/K_{k}/Cleaned/"
    subprocess.call(f"mkdir -p {outDir}", shell=True)
    column = column.replace(" ", "_")
    outDir = f"{inDir}/K_{k}/Cleaned/{column}"
    subprocess.call(f"mkdir -p {outDir}", shell=True)
    if mode == "all":
        outDir = f"{inDir}/K_{k}/Cleaned/{column}/All/"
        subprocess.call(f"mkdir -p {outDir}", shell=True)

    df = pd.read_csv(f"{inDir}/K_{k}/k_{k}.tsv", sep = "\t", index_col = 0)
    column = column.replace("_", " ")
    df = df[~pd.isna(df[column])]

    common = pd.DataFrame()
    commonexcel = pd.DataFrame()

    for keyword in keywords:
        dfKeyword = df[df[column].str.contains(keyword, case = False)]
        dfKeyword2 = df[df[column].str.contains(keyword, case = False)]
        commonexcel = pd.concat([commonexcel, dfKeyword2])
        dfKeyword = dfKeyword[dfKeyword["Condition"] != "OEvsKO"]
        dfKeyword = dfKeyword[["Z_WT", "Z_KO", "Z_OE"]]
        dfKeyword.columns = ["WT", "dH2AZ", "OEH2AZ"]
        common = pd.concat([common, dfKeyword])
        if mode != "all":
            if dfKeyword.shape[0] > 0:
                dfKeyword2["Id"] = dfKeyword2.index.tolist()
                dfKeyword2 = dfKeyword2.drop_duplicates("Id")
                dfKeyword2.pop("Id")
                dfKeyword2.to_excel(f"{outDir}/{keyword}.xlsx")
                dfKeyword["Id"] = dfKeyword.index.tolist()
                dfKeyword = dfKeyword.drop_duplicates("Id")
                dfKeyword.pop("Id")
                c = sns.clustermap(dfKeyword, row_cluster=False, col_cluster=True, yticklabels=False, cmap = "bwr")

                plt.title(f"{keyword} ({dfKeyword.shape[0]} Genes)", fontsize = 14)
                plt.savefig(f"{outDir}/{keyword}.png")
                plt.clf()
    if mode == "all":
        if common.shape[0] > 0:
            common = common.sort_values(by = "WT")
            commonexcel = commonexcel.sort_values(by = "Z_WT")
            common["Id"] = common.index.tolist()
            common = common.drop_duplicates("Id")
            common.pop("Id")
            commonexcel["Id"] = commonexcel.index.tolist()
            commonexcel = commonexcel.drop_duplicates("Id")
            commonexcel.pop("Id")
            commonexcel.to_excel(f"{outDir}/{'_'.join(keywords)}.xlsx")
            c = sns.clustermap(common, row_cluster=False, col_cluster=True, yticklabels=False, cmap = "bwr")
            plt.title(f"{keyword} ({common.shape[0]} Genes)", fontsize = 14)
            plt.savefig(f"{outDir}/{'_'.join(keywords)}.png")
            plt.clf()

def heatMapByKeyword(inDir, k, column, keywordTxt, mode, condition, direction):
    dico = {"dH2AZ" : "KO", "OEH2AZ" : "OE"}
    conditionSign = f"{dico[condition[0]]}vsWT"
    outFolder = f'{"_".join(condition)}_{direction}'
    keywords = "".join(open(keywordTxt, "r").readlines()).split("\n")
    plt.figure(figsize=(15, 10))
    print(keywords)
    print(f"{inDir}/K_{k}/k_{k}.tsv")
    outDir = f"{inDir}/K_{k}/{column}/{outFolder}"
    subprocess.call(f"mkdir -p {outDir}", shell=True)
    df = pd.read_csv(f"{inDir}/K_{k}/k_{k}.tsv", sep = "\t", index_col = 0)
    df = df[~pd.isna(df[column])]
    dfKeyword = pd.DataFrame()
    dfKeyword3 = pd.DataFrame()
    for keyword in keywords:
        dfKeyword = pd.concat([dfKeyword, df[df[column].str.contains(keyword, case = False)]])
        dfKeyword2 = pd.concat([dfKeyword, df[df[column].str.contains(keyword, case = False)]])
        dfKeyword2["Id"] = dfKeyword2.index.tolist()
        dfKeyword2 = dfKeyword2.drop_duplicates(subset = ["Id"])
        dfKeyword3 = pd.concat([dfKeyword3, dfKeyword2])
        if mode != 'all':
            dfKeyword = dfKeyword[dfKeyword["Condition"].str.contains(conditionSign)]
            dfKeyword = dfKeyword[["Z_WT", "Z_KO", "Z_OE"]]
            dfKeyword.columns = ["WT", "dH2AZ", "OEH2AZ"]
            if direction == "up":
                dfKeyword = dfKeyword[dfKeyword[condition[0]] > dfKeyword[condition[1]]]
            else:
                dfKeyword = dfKeyword[dfKeyword[condition[0]] < dfKeyword[condition[1]]]
            if dfKeyword.shape[0] > 0:
                dfKeyword["Id"] = dfKeyword.index.tolist()
                dfKeyword = dfKeyword.drop_duplicates(subset = ["Id"])
                dfKeyword.pop("Id")
                c = sns.heatmap(dfKeyword, cmap = "bwr")
                plt.title(f"{keyword} ({dfKeyword.shape[0]} Genes)", fontsize = 14)
                plt.savefig(f"{outDir}/{keyword}.png")
                dfKeyword2 = dfKeyword2[dfKeyword2["Condition"].str.contains(conditionSign)]
                dfKeyword2.to_excel(f"{outDir}/{keyword}.xlsx")
            dfKeyword = pd.DataFrame()
        plt.clf()
    if mode == "all":
        dfKeyword4 = dfKeyword3[["Z_WT", "Z_KO", "Z_OE"]]
        dfKeyword4.columns = ["WT", "dH2AZ", "OEH2AZ"]
        dfKeyword4 = dfKeyword4[condition]
        if direction == "up":
            dfKeyword4 = dfKeyword4[dfKeyword4[condition[0]] > dfKeyword4[condition[1]]]
        else:
            dfKeyword4 = dfKeyword4[dfKeyword4[condition[0]] < dfKeyword4[condition[1]]]
        if dfKeyword4.shape[0] > 0:
            dfKeyword4["Id"] = dfKeyword4.index.tolist()
            dfKeyword4 = dfKeyword4.drop_duplicates(subset = ["Id"])
            dfKeyword4.pop("Id")
            c = sns.heatmap(dfKeyword4, cmap = "bwr", yticklabels=False)
            plt.title(f"{'_'.join(keywords)} ({dfKeyword4.shape[0]} Genes)", fontsize = 10)
            print(f"{outDir}/{'_'.join(keywords)}.png")
            plt.savefig(f"{outDir}/{'_'.join(keywords)}.png")
            dfKeyword4.to_csv(f"{outDir}/{'_'.join(keywords)}.tsv", sep = "\t", index=None)

def goenrichment(path, kTsv):
    df = pd.read_csv(kTsv, sep = "\t", index_col = 0)
    colors = clusterColors(df)
    files = glob.glob(f"{path}/*GOen*.tsv")
    for f in files:
        go = pd.read_csv(f, sep = "\t")
        for index, row in go.iterrows():
            genes = row["Result gene list"].split(",")
            genes.pop()
            subDf = df[df.index.isin(genes)]
            name = row["Name"].replace('/', '_')
            clusterMapGO(subDf, path, colors, name)

def searchInGoEnrichment(path, kTsv, toFind):
    dico = {}
    mode = ["OE", "KO"]
    go = pd.read_csv(toFind, sep = "\t")
    go = go.fillna("")
    for m in mode:
        dico[m] = {}
        for d in ["Up", "Down"]:
            dico[m][d] = {}
            print(f"{path}/{m}vsWT/{d}/*Goen*.tsv")
            for f in glob.glob(f"{path}/{m}vsWT/{d}/*Goen*.tsv"):
                e = os.path.basename(f).split("_")[-1].split(".")[0]
                dico[m][d][e] = {}
                for col in go.columns.tolist():
                    revigo = pd.read_csv(f, sep = "\t")
                    dico[m][d][e] = revigo[revigo["ID"].isin(go[col].tolist())].shape[0]
    print(dico)


def heatMapPerChrom(resultFile, gtf, bin):
    dico = defaultdict(list)
    df = pd.read_csv(resultFile, sep = "\t")
    gtf = pd.read_csv(gtf, comment = "#", sep = "\t", header = None)
    for index, row in gtf.iterrows():
        values = row[8].split(";")
        for value in values:
            if "FGRAMPH1_" in value:
                dico["Id"].append(value.split(" ")[-1][1:-1])
                dico["Chrom"].append(row[0])
                dico["Start"].append(row[3])
                dico["End"].append(row[4])
    final = pd.DataFrame(data = dico)
    final["Id"] = final["Id"].str.replace("T", "G")
    final["Length"] = final["End"] - final["Start"]
    merged = final.merge(df, on = ["Id"])
    for chrom in set(merged["Chrom"]):
        zscore = []
        chromDf = merged[merged["Chrom"] == chrom]
        indexes = sepPerBin(chromDf, bin)
        for indexe in indexes:
            dico = defaultdict(list)
            subDf = chromDf[chromDf.index.isin(indexe)]
            for col in subDf.columns.tolist():
                if "norm." in col:
                    dico[col].append(subDf[col].mean())
            subDf = pd.DataFrame(data = dico)
            if zScore(subDf).shape[0] != 0:
                zscore.append(zScore(subDf)["Z_WT"].tolist()[0])
        ax = sns.heatmap([zscore], cmap = "bwr", yticklabels=False, xticklabels=False, cbar = False)
        plt.savefig(f"{os.path.dirname(resultFile)}/chrom_{chrom}_{bin}.png")
        plt.clf()

def sepPerBin(df, maxBin):
    bin = 0
    indexes = []
    bins = []
    for i in df.index.tolist():
        if bin <= maxBin:
            bin += df[df.index.isin([i])]["Length"].tolist()[0]
            indexes.append(i)
        else:
            bins.append(indexes)
            bin = 0
            indexes = []
    return bins




if __name__ == "__main__":
    # heatMapPerChrom("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/KOvsWT/KOvsWT.complete.txt", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Transcript-annotation-5-uncollapsed.gtf", 10000)
    # heatMapPerChrom("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/KOvsWT/KOvsWT.complete.txt", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Transcript-annotation-5-uncollapsed.gtf", 25000)
    # heatMapPerChrom("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/KOvsWT/KOvsWT.complete.txt", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Transcript-annotation-5-uncollapsed.gtf", 50000)
    # goenrichment("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/GOEnrichment/OEvsWT/Down/", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign/K_7/k_7.tsv")
    # goenrichment("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/GOEnrichment/KOvsWT/Down/", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign/K_7/k_7.tsv")
    # goenrichment("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/GOEnrichment/OEvsWT/Up/", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign/K_7/k_7.tsv")
    # goenrichment("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/GOEnrichment/KOvsWT/Up/", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign/K_7/k_7.tsv")
    # main("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/GO/GO.complete.tsv", [], ["log2FoldChange", "padj"])
    # heatMapByKeyword("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Process", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/filamentous.txt", "all", ["dH2AZ", "WT"], "up")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Process", "process.txt", "blabla", ["dH2AZ", "WT"], "up")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/Solo/process.txt", "")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/Solo/pigmentation.txt", "")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/expression.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/filamentous.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/response.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/new_response.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Processes", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Process/All/translation.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Components", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Component/Solo/component.txt", "")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Components", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Component/Solo/membrane.txt", "")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Components", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Component/All/chromosome.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Components", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Component/All/dna.txt", "all")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Computed GO Functions", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Function/Solo/function.txt", "")
    # heatMapByKeywordAll("/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/DiffDEseq2/Résultats/OnlySign", 7, "Product Description", "/home/aurelie/Bureau/RNASeq_Scripts/RNASeq/Txt/Product/Solo/cutinase.txt", "")
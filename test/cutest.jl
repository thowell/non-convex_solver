using CUTEst

CUTEstAll = (
    "ARGLCLE",
    "BT1",
    "BT2",
    "BT3",
    "BT4",
    "BT5",
    "BT6",
    "BT7",
    "BT8",
    "BT9",
    "BT10",
    "BT11",
    "BT12",
    "BYRDSPHR",
    "COOLHANS",
    "DIXCHLNG",
    "EIGENA2",
    "EIGENACO",
    "EIGENB2",
    "EIGENBCO",
    "ELEC",
    "GRIDNETE",
    "GRIDNETH",
    "HS6",
    "HS7",
    "HS8",
    "HS9",
    "HS26",
    "HS27",
    "HS28",
    "HS39",
    "HS40",
    "HS42",
    "HS46",
    "HS47",
    "HS48",
    "HS49",
    "HS50",
    "HS51",
    "HS52",
    "HS56",
    "HS61",
    "HS77",
    "HS78",
    "HS79",
    "LCH",
    "LUKVLE1",
    "LUKVLE3",
    "LUKVLE6",
    "LUKVLE7",
    "LUKVLE8",
    "LUKVLE9",
    "LUKVLE10",
    "LUKVLE13",
    "LUKVLE14",
    "LUKVLE16",
    "LCH",
    "MSS1",
    "MARATOS",
    "MWRIGHT",
    "ORTHRDM2",
    "ORTHRDS2",
    "ORTHREGA",
    "ORTHREGB",
    "ORTHREGC",
    "ORTHREGD",
    "WOODSNE")

function sortCUTEst(probs=CUTEstAll)
    # Categorize the problems
    smallProbs = String[]
    interProbs = String[]
    largeProbs = String[]
    rightForm = Bool[]
    CUTEsizes = Tuple{Int,Int}[]
    for prob in probs
        nlp = CUTEstModel(prob)
        n,m = nlp.meta.nvar, nlp.meta.ncon
        if nlp.meta.lcon == nlp.meta.ucon
            push!(rightForm, true)
            if n < 100
                push!(smallProbs, prob)
            elseif n < 1000
                push!(interProbs, prob)
            else
                push!(largeProbs, prob)
            end
        else
            push!(rightForm, false)
        end
        push!(CUTEsizes, (n,m))
        finalize(nlp)
    end
    CUTEstProbs = Dict(:all => CUTEstAll, :small => smallProbs, :medium => interProbs,
        :large => largeProbs, :sizes => CUTEsizes, :form => rightForm)
end

smallProbs = [
    "BT1", "BT2", "BT3", "BT4", "BT5", "BT6", "BT7", "BT8", "BT9", "BT10", "BT11", "BT12",
    "BYRDSPHR", "COOLHANS", "DIXCHLNG", "HS6", "HS7", "HS8", "HS9", "HS26", "HS27", "HS28",
    "HS39", "HS40", "HS42", "HS46", "HS47", "HS48", "HS49", "HS50", "HS51", "HS52", "HS56",
    "HS61", "HS77", "HS78", "HS79", "MSS1", "MARATOS", "MWRIGHT", "ORTHREGB"
]
interProbs = ["ARGLCLE", "ELEC"]
largeProbs = [
    "EIGENA2", "EIGENACO", "EIGENB2", "EIGENBCO", "GRIDNETE", "GRIDNETH", "LCH", "LUKVLE1",
    "LUKVLE3", "LUKVLE6", "LUKVLE7", "LUKVLE8", "LUKVLE9", "LUKVLE10", "LUKVLE13", "LUKVLE14",
    "LUKVLE16", "LCH", "ORTHRDM2", "ORTHRDS2", "ORTHREGA", "ORTHREGC", "ORTHREGD", "WOODSNE"
]
CUTEsizes = [
    (200, 399), (2, 1), (3, 1), (5, 3), (3, 2), (3, 2), (5, 2), (5, 3), (5, 2),
    (4, 2), (2, 2), (5, 3), (5, 3), (3, 2), (9, 9), (10, 5), (2550, 1275), (2550, 1275),
    (2550, 1275), (2550, 1275), (600, 200), (7564, 3844), (7564, 3844), (2, 1), (2, 1),
    (2, 2), (2, 1), (3, 1), (3, 1), (3, 1), (4, 2), (4, 3), (4, 2), (5, 2), (5, 3), (5, 2),
    (5, 2), (5, 3), (5, 3), (5, 3), (7, 4), (3, 2), (5, 2), (5, 3), (5, 3), (3000, 1),
    (10000, 9998), (10000, 2), (9999, 4999), (10000, 4), (10000, 9998), (10000, 6),
    (10000, 9998), (9998, 6664), (9998, 6664), (9997, 7497), (3000, 1), (90, 73), (2, 1),
    (5, 3), (8003, 4000), (5003, 2500), (8197, 4096), (27, 6), (5005, 2500), (5003, 2500),
    (4000, 3001)
]
CUTEstProbs = Dict(:all => CUTEstAll, :small => smallProbs, :medium => interProbs,
    :large => largeProbs, :sizes => CUTEsizes, :form => rightForm)

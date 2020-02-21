#Estimation of the mAb glycoprofile based on NSD model simulation data from the B&B paper
#The model will be trained in 4 experiments (control, 10G, 10G20U, 50G5U)
#The model will be tuned (number of neurons) to simulate the 10G5U experiment

#preliminary results
#HL1=18
#HL2=34
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#normalization factors for the glycans
error = np.zeros([41,41])

epochs = 20000

for i in range(0, 41):
    print (i)
    for j in range(0, 41):

        np.random.seed(0)
        feature_set = np.array([[0.0183726991627284,	0.01650406552988,	0.01520794246292,	0.0148381782758314,	0.0558761646117539,	0.171630894636261,	0.103103241845479,	0.469622154421602,	0.247119408926328,	0.420284595449271,	0.5],
        [0.0826343574907975,	0.0897889336838478,	0.0935260559530952,	0.0964033367226371,	0.108012453116654,	0.12092554295931,	0.130550181669173,	0.5,	0.229403223439735,	0.336326704479673,	0.458363056156244],
        [0.388891391700973,	0.374659766473817,	0.340880574687301,	0.289369063398563,	0.422528750628343,	0.443831363104643,	0.452869530618517,	0.457075771624724,	0.436053711869045,	0.474554690413508,	0.5],
        [0.423337234394362,	0.414423141055418,	0.389286295630511,	0.349279215172575,	0.447023473278268,	0.463019056650955,	0.469516768128567,	0.470653121041838,	0.456102769260068,	0.483143191458242,	0.5],
        [0.370840602299719,	0.414427854419705,	0.459357055553125,	0.482043564674911,	0.370912092000121,	0.414541133526578,	0.459479173804931,	0.388946497786973,	0.386967152471952,	0.431986250201949,	0.477291987405235],
        [0.024999432463772,	0.0622549975938363,	0.133846231937804,	0.180343810599071,	0.0251259571054356,	0.0630064426821092,	0.135281443588675,	0.213003411753867,	0.127224002250291,	0.254621162041344,	0.415291685278284],
        [0.0137357899397877,	0.0512383449278091,	0.109140244950048,	0.147762386926989,	0.0140475427401576,	0.0522814186746003,	0.11098227523919,	0.195923951777435,	0.0889895778890605,	0.218522579192886,	0.400688604726326]]).T

        
        labels = np.array([[0.064129641,	0.060006061,	0.057963183,	0.060578699,	0.04801502,	0.056749127,	0.05056766,	0.03729378,	0.041926173,	0.03140784,	0.036667007],
        [0.524051425,	0.515046527,	0.520968923,	0.536518635,	0.4891363,	0.484554236,	0.475295968,	0.473100902,	0.418039198,	0.443201672,	0.41417279],
        [0.346344857,	0.363193137,	0.340245869,	0.342065591,	0.3982028,	0.380502521,	0.39632325,	0.391301779,	0.418565498,	0.41475993,	0.417065795],
        [0.004505264,	0.004505264,	0.00285,	0.004378582,	0.004378582,	0.004378582,	0.00486606,	0.00486606,	0.008383955,	0.00375555,	0.006653057],
        [0.046663351,	0.050148329,	0.04065236,	0.047542788,	0.05807403,	0.034639551,	0.065576103,	0.072536927,	0.079412147,	0.074366598,	0.083883881],
        [0.018810726,	0.007100683,	0.034616107,	0.008915705,	0.00657189,	0.043554566,	0.007370959,	0.044413505,	0.029021069,	0.03011102,	0.0345]]).T
        
#ANN parameters
        wh1 = np.random.rand(len(feature_set[0]),i)
        wh2 = np.random.rand(i, j)
        wo  = np.random.rand(j, len(labels[0]))
        lr = 0.5

        #print (wh1) #it was silenced in order to save some time during optimization

        for epoch in range(epochs):
            #FEEDFORWARD
            zh1 = np.dot(feature_set, wh1)
            ah1 = sigmoid(zh1)

            zh2 = np.dot(ah1, wh2)
            ah2 = sigmoid(zh2)
            
            zo = np.dot(ah2, wo)
            ao = sigmoid(zo)

#ANN error
            error_output = ((1/2)*(np.power((ao - labels), 2)))

#Part1: from HL2 to the Output
            dcost_dao = ao - labels
            dao_dzo   = sigmoid_der(zo)
            dzo_dwo   = ah2
            dcost_dwo = np.dot(dzo_dwo.T, dcost_dao*dao_dzo)
#Part2: from HL1 to HL2
            dcost_dzo = dcost_dao*dao_dzo
            dzo_dah2  = wo
            dcost_dah2 = np.dot(dcost_dzo, dzo_dah2.T)
            dah2_dzh2 = sigmoid_der(zh2)
            dzh2_dwh2 = ah1
            dcost_dwh2 = np.dot(dzh2_dwh2.T, dah2_dzh2*dcost_dah2)
#Part2: from the Input to HL1
            dcost_dzh2 = dcost_dah2*dah2_dzh2
            dzh2_dah1  = wh2
            dcost_dah1 = np.dot(dcost_dzh2, dzh2_dah1.T)
            dah1_dzh1 = sigmoid_der(zh1)
            dzh1_dwh1 = feature_set
            dcost_dwh1 = np.dot(dzh1_dwh1.T, dah1_dzh1*dcost_dah1)
#update weights
            wo  -= lr*dcost_dwo
            wh2 -= lr*dcost_dwh2
            wh1 -= lr*dcost_dwh1

        single_point = np.array([[0.191748143223755,	0.312113968081812,	0.380803692849343,	0.409882994978949],
        [0.224617604609086,	0.330026187569855,	0.450128136934431,	0.499273729996663],
        [0.436103830681038,	0.46701285663533,	0.489039004554347,	0.482360295557279],
        [0.456225950784437,	0.478147463538269,	0.492930372387265,	0.489076653811823],
        [0.386966644364922,	0.431988896193274,	0.477296454912327,	0.5],
        [0.127242617709794,	0.254714151191054,	0.415434213492293,	0.5],
        [0.0890932949562721,	0.218677770890606,	0.400698134556139,	0.5]]).T

        
        #results
        resulth1 = sigmoid(np.dot(single_point, wh1))
        resulto1 = sigmoid(np.dot(resulth1, wh2))
        resulto2 = sigmoid(np.dot(resulto1, wo))

        experimental_results = np.array([[0.0521049,	0.049748503,	0.051655478,	0.047616816],
        [0.447665056,	0.418781566,	0.46057572,	0.397471592],
        [0.415573959,	0.418381974,	0.407668677,	0.438894841],
        [0.007361399,	0.008429525,	0.007509387,	0.00345],
        [0.072348746,	0.078123761,	0.069063602,	0.089123178],
        [0.00494594,	0.0264,	0.003527136,	0.021875569]]).T
      

        error_table = abs(resulto2 - experimental_results)
        
        error[i,j] = error_table.sum()

for i in range(0,41):
    for j in range(0,41):
        if error[i,j] == error.min():
            x = i
            y = j

print ("The minimum error is found in the following coordinates (number of nodes): x=", x, ", y=", y)
print ("The error of the optimized NN is: ", error[x,y])
print ("The optimum nodes combination therefore is: HL1: ", x, ", HL2: ", y)


print (error)


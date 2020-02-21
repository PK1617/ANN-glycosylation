#Estimation of the Asn-745 of the Fc-EPO protein glycoprofile for 4 GalT KO, using training sets of 3KOs at a time (from 2019 Nicole Borth paper)

#HL1:20
#HL2:11

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

epochs = 20000
max_nodes = 30
error = np.zeros([max_nodes,max_nodes])

for i in range(1, max_nodes+1):
    print (i)
    for j in range(1, max_nodes+1):

        np.random.seed(0)
        feature_set = np.array([[0.364963503649635,	0.344160583941606,	0.387773722627737,	0.000072992700729927,	3.64963503649635E-05,	0.00010948905109489,	0.000072992700729927,	0.00010948905109489,	0.000328467153284672,	3.64963503649635E-05,	3.64963503649635E-05,	3.64963503649635E-05],
        [0.364963503649635,	0.00113138686131387,	0.00164233576642336,	1,	0.836532846715328,	0.771094890510949,	0.000583941605839416,	0.0012043795620438,	0.00591240875912409,	0.0022992700729927,	0.00208029197080292,	0.00167883211678832],
        [0.364963503649635,	0.000072992700729927,	0.000182481751824818,	3.64963503649635E-05,	3.64963503649635E-05,	0,	0.567737226277372,	0.255401459854015,	0.61485401459854,	0,	0,	0],
        [0.364963503649635,	0,	0.000218978102189781,	0,	3.64963503649635E-05,	0,	0,	0,	0,	0.255839416058394,	0.401459854014599,	0.417043795620438]]).T

        labels = np.array([[0.034,	0.017,	0.046,	0.016,	0,	0.014,	0.036,	0.018,	0.022,	0.037,	0.018,	0.018],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.12,	0.105,	0.127,	0.019,	0.013,	0.013,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0.056,	0.055,	0.058,	0,	0,	0,	0.014,	0.019,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0.187,	0.19,	0.202,	0.289,	0.278,	0.3,	0.339,	0.325,	0.336],
        [0,	0,	0,	0.045,	0.047,	0.041,	0.101,	0.065,	0.086,	0.101,	0.097,	0.094],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.218,	0.231,	0.251,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.015,	0.013,	0.016,	0.063,	0.074,	0.069,	0.021,	0.021,	0.021,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.05,	0.063,	0.069,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.169,	0.133,	0.136,	0.013,	0.013,	0,	0,	0,	0,	0,	0,	0],
        [0,	0.016,	0,	0.097,	0.097,	0.103,	0.016,	0.018,	0.016,	0,	0,	0],
        [0,	0,	0,	0.011,	0.015,	0.015,	0,	0,	0,	0,	0,	0],
        [0.01,	0,	0,	0.301,	0.295,	0.291,	0.371,	0.455,	0.399,	0.411,	0.376,	0.405],
        [0,	0,	0,	0.067,	0.061,	0.059,	0.103,	0.084,	0.094,	0.068,	0.12,	0.103],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.284,	0.283,	0.232,	0.026,	0.03,	0.028,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0.072,	0.082,	0.077,	0.031,	0.034,	0.033,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.079,	0.106,	0.087,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0.021,	0.032,	0.035,	0.027,	0.029,	0.028,	0.031,	0.026,	0.028,	0.031,	0.045,	0.044]]).T


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

            #HL2
            #Phase 1
            #error_out2 = ((1/2)*(np.power((ao2 - labels), 2)))

            #dcost_dao2 = ao2 - labels
            #dao_dzo2 = sigmoid_der(zo2)
            #dzo_dwo2 = ah2
            #dcost_wo2 = np.dot(dzo_dwo2.T, dcost_dao2*dao_dzo2)

            #Phase 2
            #dcost_dzo2 = dcost_dao2 * dao_dzo2
            #dzo_dah2 = wo2
            #dcost_dah2 = np.dot(dcost_dzo2, dzo_dah2.T)
            #dah_dzh2 = sigmoid_der(zh2)
            #dzh_dwh2 = ah1
            #dcost_wh2 = np.dot(dzh_dwh2.T, dah_dzh2*dcost_dah2)

            #Phase 3
            #dcost_dzo1 = dcost_dao1 * dao_dzo1
            #dzo_dah1 = wo1
            #dcost_dah1 = np.dot(dcost_dzo1, dzo_dah1.T)
            #dah_dzh1 = sigmoid_der(zh1)
            #dzh_dwh1 = feature_set
            #dcost_wh1 = np.dot(dzh_dwh1.T, dah_dzh1*dcost_dah1)

        single_point = np.array([0.000833333333333333, 0.0031, 0.0002, 0])

        #results
        resulth1 = sigmoid(np.dot(single_point, wh1))
        resulto1 = sigmoid(np.dot(resulth1, wh2))
        resulto2 = sigmoid(np.dot(resulto1, wo))
        #print (resulto2) #it was silenced in order to save some time during optimization

        experimental_results = np.array([0.03,	0,	0,	0,	0,	0,	0,	0,	0.351,	0.0716666666666667,	0,	0,	0,	0,	0,	0,	0,	0,	0.432333333333333,	0.0746666666666667,	0,	0,	0,	0,	0,	0.04])

        error_table = abs(resulto2 - experimental_results)
        
        error[i-1,j-1] = error_table.sum()

#print (error)


for i in range(1,max_nodes):
    for j in range(1,max_nodes):
        if error[i,j] == error.min():
            x = i
            y = j

print ("The minimum error is found in the following coordinates (number of nodes): x=", x, ", y=", y)
print ("The error of the optimized NN is: ", error[x,y])
print ("The optimum nodes combination therefore is: HL1: ", x+1, ", HL2: ", y+1)




import numpy as np

def sigmond(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmond(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork():
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmond(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmond(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmond(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1
    
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_hl = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmond(sum_hl)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmond(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w2 * h2 + self.b3
                o1 = sigmond(sum_o1)
                y_pred = o1

                #подсчёт частных производных L от w1
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон ol
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 =  deriv_sigmoid(sum_o1)
                d_ypred_d_hl = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                # Нейрон hl
                d_h1_d_w1 =  x[0] * deriv_sigmoid(sum_hl)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_hl)
                d_h1_d_b1 = deriv_sigmoid(sum_hl)
                
                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем вес и смещения
                # Нейрон h1
                self.w1 = learn_rate * d_L_d_ypred * d_ypred_d_hl * d_h1_d_w1
                self.w2 = learn_rate * d_L_d_ypred * d_ypred_d_hl * d_h1_d_w2
                self.b1 = learn_rate * d_L_d_ypred * d_ypred_d_hl * d_h1_d_b1
                
                # Нейрон h2
                self.w3 = learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 = learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 = learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                # Нейрон o1
                self.w5 = learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 = learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 = learn_rate * d_L_d_ypred * d_ypred_d_b3


# --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                print(learn_rate * d_L_d_ypred * d_ypred_d_hl * d_h1_d_w1)
                print('w1=', self.w1)
                print('w2=', self.w2)
                print('bl=', self.b1)
                print('w3=', self.w3)
                print('w4=', self.w4)
                print('b2=', self.b2)
                print('w5=', self.w5)
                print('w6=', self.w6)
                print('b3=', self.b3)
      
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])

all_y_trues = np.array([
    1,
    0,
    0,
    1,
])

network = OurNeuralNetwork()
network.train(data, all_y_trues)

emily = np.array([-7, -3]) # 128 фунтов, 63 дюйма
frank = np.array([20, 2]) # 155 фунтов, 68 дюймов
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))

# main.py
from michigrad.visualize import show_graph
from michigrad.nn import MLP, Layer, Tanh

def main():
    xor = MLP([
        Layer(2, 2),
        Tanh(),
        Layer(2, 1)
    ])

    xs = [[0,0],[0,1],[1,0],[1,1]] 
    ys = [0,1,1,0]

    yhats = [xor(x) for x in xs]

    loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

    dot = show_graph(loss)
    dot.render("images/graphForwardP3", view=True)

    loss.backward() 

    dot = show_graph(loss)
    dot.render("images/graphBackwardP3", view=True)

    for _ in range(30):
        yhats = [xor(x) for x in xs]

        loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

        for p in xor.parameters():
            p.grad = 0.

        loss.backward()

        for p in xor.parameters():
            p.data -= p.grad * 0.01

        print(loss)

    for _ in range(1000):
        yhats = [xor(x) for x in xs]

        loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

        for p in xor.parameters():
            p.grad = 0.

        loss.backward()

        for p in xor.parameters():
            p.data -= p.grad * 0.01


    print("Despues de 1000 iteraciones: ",loss)
    dot = show_graph(loss)
    dot.render("images/graphFinalP3", view=True)

if __name__ == "__main__":
    main()

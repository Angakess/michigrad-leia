# main.py
from michigrad.visualize import show_graph
from michigrad.nn import MLP, Layer

def main():
    xor = MLP([Layer(2,2)])

    xs = [[0,0],[0,1],[1,0],[1,1]] 
    ys = [0,1,1,0]

    yhats = [xor(x)[0] for x in xs]

    loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

    dot = show_graph(loss)
    dot.render("images/graphForwardP1", view=True)

    loss.backward() 

    dot = show_graph(loss)
    dot.render("images/graphBackwardP1", view=True)

    for _ in range(30):
        yhats = [xor(x)[0] for x in xs]

        loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

        for p in xor.parameters():
            p.grad = 0.

        loss.backward()

        for p in xor.parameters():
            p.data -= p.grad * 0.01

        print(loss)

    for _ in range(1000):
        yhats = [xor(x)[0] for x in xs]

        loss = sum([(y - yhat)**2 for y, yhat in zip(ys,yhats)])/4

        for p in xor.parameters():
            p.grad = 0.

        loss.backward()

        for p in xor.parameters():
            p.data -= p.grad * 0.01

    print("Despues de 1000 iteraciones: ",loss)
    dot = show_graph(loss)
    dot.render("images/graphFinalP1", view=True)

if __name__ == "__main__":
    main()

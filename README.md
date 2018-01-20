# GradientDescent
-------------------

<a href = "https://en.wikipedia.org/wiki/Gradient_descent">Gradient Descent</a> is one of `the most popular and important` optimization Algorithm in the Machine Learning. This Algorithm's Applications can be seen far wide in the Deep Learning.

Gradient descent work's in a way that for each iteration it minizies the error function or the loss function caused due to the prediction.

This Presentation is an attempt to demonstrat the basic use `Impelementation` of gradient descent in Linear Regression.

![gd](https://github.com/mattnedrich/GradientDescentExample/blob/master/gradient_descent_example.gif)


# Project

- Here I've considered few point's with a random noise to it.
![gd](https://github.com/Rupii/GradientDescent/blob/master/images/data.png?raw=true)

- as you can see we can fit a line throgh the data. 

### Y = m * X + b
+ is all we need

- we have X, Y as our data and we need to find the parameters m, b so that we can fit a Regression Line through it.
```{pthon}
initial_m = 0
initial_b = 0
# initially we can assign any random values m, b 
# preferablly a random number's from gaussian destirbution.
```
- our plot look's like this when we pass `y = m * x + b` with initial_m, initial_b ![gd](https://github.com/Rupii/GradientDescent/blob/master/images/initial_plot.png?raw=true)

- so look's like that could be worst fit possible
- let's calicaulate the loss function or error function .
that can be don by ![error](https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png)
the above is Squared Mean error for `y = mX + b`  Yi is the i'th y of the data set

Gradient Descent
-------------------


Now's comes the derivative part.

In order to compute the best fit we run gradient descent for  `all the parameters` m, b in this case

there is just one simple equation for that, consider theta as parameter and it goes something like ![gd](https://www.codeproject.com/KB/recipes/879043/GradientDescent.jpg)


in this case we need only need derivate's for two parametes m and b ![gd](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)
 `where N is the No. of obseravation's`

# Learning Rate
- Learning is a random value one need to assign in order to find the opimal m, b
- Learning Rate determines how quickly your algorithm finds the optimal values.
 There is no standard Learning rate, it is somthing one need to develop by practising as many problems as possible.
 
 Gradient descent code
 ---------------------------
 ```{python}
 
learning_rate = 0.001

loss = list()

plt.plot(X, y, "*")

for i in range(200):
    """
    caliculating the darivative with respect to m and b 
    chosing learning rate as 0.001 running gradient descent for twenty iterations
    appling the learning rate changing the parameter gives the perfect fit
    
    """
    # computing the derivatives
    dm = (X.dot(y)).mean() - (m * ( X.dot(X) ).mean() + b * X.mean())
    db = y.mean() - (m * X.mean() + b)

    # appling gradient descent with learning rate
    t1 = m - learning_rate * (-2/20) * (dm) 
    t2 = b - learning_rate * db * (-2/20)

    # updating the parameters
    m = t1
    b = t2
    
    yhat = m * X + c
    
    error = ((y - (m * X + b) )**2).sum()
    loss.append(error)
    plt.plot(X, yhat, alpha= 0.3 )
```
![output](https://github.com/Rupii/GradientDescent/blob/master/images/gradient.png?raw=true)

here you can how the gradient descent `converges ` towards the optimal solution.

# final Y = m * X + b

our final plot is ![out](https://github.com/Rupii/GradientDescent/blob/master/images/final_gradient.png?raw=true)

# cost function

for every iteration of the gradient descent the error rate keeps decressing
![cost](https://github.com/Rupii/GradientDescent/blob/master/images/loss.png?raw=true)
you can see after some iteration it is almost constant that states that it is the `optimum` it cannot optimize futher than that

# Thank You
that's all there is to it.

for complete code click <a href = "https://github.com/Rupii/GradientDescent/blob/master/Simple%20linear%20Regression.ipynb">Here</a> 
for more on Deep Learning 
visit my blog <a href = "www.rupi.ml"> rupi.ml </a>

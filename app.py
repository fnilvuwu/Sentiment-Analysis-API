from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64
import time

app = FastAPI()

# Function to calculate D_AB
def calculate_D_AB(Xa, a_AB, a_BA, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T):
    Xb = 1 - Xa  # Mole fraction of B
    D = Xa*(D_BA0) + Xb*np.log(D_AB0) + \
        2*(Xa*np.log(Xa+(Xb*λ_b)/λ_a)+Xb*np.log(Xb+(Xa*λ_a)/λ_b)) + \
        2*Xa*Xb*((λ_a/(Xa*λ_a+Xb*λ_b))*(1-(λ_a/λ_b)) +
                 (λ_b/(Xa*λ_a+Xb*λ_b))*(1-(λ_b/λ_a))) + \
        Xb*q_a*((1-((Xb*q_b*np.exp(-a_BA/T))/(Xa*q_a+Xb*q_b*np.exp(-a_BA/T)))**2)*(-a_BA/T)+(1-((Xb*q_b)/(Xb*q_b+Xa*q_a*np.exp(-a_AB/T)))**2)*np.exp(-a_AB/T)*(-a_AB/T)) + \
        Xa*q_b*((1-((Xa*q_a*np.exp(-a_AB/T))/(Xa*q_a*np.exp(-a_AB/T)+Xb*q_b))**2)*(-a_AB/T)+(1-((Xa*q_a)/(Xa*q_a+Xb*q_b*np.exp(-a_BA/T)))**2)*np.exp(-a_BA/T)*(-a_BA/T))
    return np.exp(D)

# Objective function for minimization
def objective(params, Xa_values, D_AB_exp, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T):
    a_AB, a_BA = params
    D_AB_calculated = calculate_D_AB(Xa_values, a_AB, a_BA, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
    return np.sum((D_AB_calculated - D_AB_exp)**2)

@app.get("/", response_class=HTMLResponse)
def input_data():
    return """
    <form action="/calculate" method="post">
        <label for="D_AB_exp">D_AB_exp:</label><br>
        <input type="text" id="D_AB_exp" name="D_AB_exp"><br>
        <label for="T">Temperature (T):</label><br>
        <input type="text" id="T" name="T"><br>
        <label for="Xa">Mole Fraction of A (Xa):</label><br>
        <input type="text" id="Xa" name="Xa"><br>
        <label for="λ_a">λ_a:</label><br>
        <input type="text" id="λ_a" name="λ_a"><br>
        <label for="λ_b">λ_b:</label><br>
        <input type="text" id="λ_b" name="λ_b"><br>
        <label for="q_a">q_a:</label><br>
        <input type="text" id="q_a" name="q_a"><br>
        <label for="q_b">q_b:</label><br>
        <input type="text" id="q_b" name="q_b"><br>
        <label for="D_AB0">D_AB0:</label><br>
        <input type="text" id="D_AB0" name="D_AB0"><br>
        <label for="D_BA0">D_BA0:</label><br>
        <input type="text" id="D_BA0" name="D_BA0"><br><br>
        <input type="submit" value="Calculate">
    </form>
    """

@app.post("/calculate", response_class=HTMLResponse)
def calculate(D_AB_exp: float = Form(...), T: float = Form(...), Xa: float = Form(...), 
              λ_a: str = Form(...), λ_b: str = Form(...), q_a: float = Form(...), 
              q_b: float = Form(...), D_AB0: float = Form(...), D_BA0: float = Form(...)):
    
    λ_a = eval(λ_a)
    λ_b = eval(λ_b)
    
    # Initial parameters
    params_initial = [0, 0]
    tolerance = 1e-12
    max_iterations = 1000
    iteration = 0

    start_time = time.time()

    while iteration < max_iterations:
        result = minimize(objective, params_initial, args=(Xa, D_AB_exp, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T), method='Nelder-Mead')
        a_AB_opt, a_BA_opt = result.x
        D_AB_opt = calculate_D_AB(Xa, a_AB_opt, a_BA_opt, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
        error = np.abs(D_AB_opt - D_AB_exp)
        if np.max(np.abs(np.array(params_initial) - np.array([a_AB_opt, a_BA_opt]))) < tolerance:
            break
        params_initial = [a_AB_opt, a_BA_opt]
        iteration += 1

    execution_time = time.time() - start_time

    Xa_values = np.linspace(0, 0.7, 100)
    D_AB_values = calculate_D_AB(Xa_values, a_AB_opt, a_BA_opt, λ_a, λ_b, q_a, q_b, D_AB0, D_BA0, T)
    
    plt.plot(Xa_values, D_AB_values)
    plt.xlabel('Fraction molaire de A')
    plt.ylabel('Coefficient de diffusion (cm^2/s)')
    plt.title('Variation du coefficient de diffusion en fonction du fraction molaire')
    plt.grid(True)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return f"""
    <h1>Results</h1>
    <p>a_AB_opt: {a_AB_opt}</p>
    <p>a_BA_opt: {a_BA_opt}</p>
    <p>D_AB_opt: {D_AB_opt}</p>
    <p>Error: {error}</p>
    <p>Iterations: {iteration}</p>
    <p>Execution Time: {execution_time} seconds</p>
    <img src="data:image/png;base64,{graph}" />
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


import streamlit as st
from engine import Run_Monte_Carlo_Simulation, return_weights
import numpy as np
import pandas as pd

st.title(" Portfolio Monte Carlo Simulation")
st.markdown("---")
st.markdown("""
This application allows you to perform a Monte Carlo simulation for portfolio optimization.
""")

sidebar=st.sidebar
with sidebar:
    days=st.number_input("Trading Days",min_value=1,step=1,max_value=365,value=252)
    
    st.markdown("---")

    assets=st.number_input("Assets",min_value=1,step=1,max_value=10,value=5)
    
    st.markdown("---")

    number_of_portfolios=st.number_input("Portfolios",min_value=1,step=1,max_value=10000,value=1000)

    
    st.markdown("---")

    risk_free_rate=st.number_input("Risk-Free Rate",min_value=0.0,step=0.01,max_value=1.0,value=0.05)

    
    st.markdown("---")

    if st.button("Run Simulation"):
        st.success("Simulation Running...")

    monte_carlo_results =  Run_Monte_Carlo_Simulation(days, assets, number_of_portfolios, risk_free_rate)
    portfolio_returns, portfolio_volatility, sharpe_ratio, Risk_metrics, max_sharpe_idx = monte_carlo_results



st.markdown("## Max Sharpe Index")
st.metric(label="Max Sharpe Index", value=f"{np.argmax(sharpe_ratio)}")

st.markdown("## Metrics Display")
st.markdown(f"**Portfolio Returns** ")
st.metric(label="Portfolio Returns", value=f"{portfolio_returns[max_sharpe_idx]:.2f}")  
st.markdown("---")
st.markdown(f"**Portfolio Volatility**")
st.metric(label="Portfolio Volatility", value=f"{portfolio_volatility[max_sharpe_idx]:.2f}")
st.markdown("---")
st.markdown(f"**Sharpe Ratio**")
st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio[max_sharpe_idx]:.2f}")
st.markdown("---")
st.markdown(f"**Risk Metrics (5th Percentile)**")
st.metric(label="Risk Metrics", value=f"{Risk_metrics:.2f}")
st.markdown("---")
    

st.markdown("## Visualization  scatterplot")
st.markdown("### Scatterplot of Portfolio Returns vs Portfolio Volatility")
st.scatter_chart(data={"Portfolio Returns": portfolio_returns, "Portfolio Volatility": portfolio_volatility})


return_weights = return_weights(assets, number_of_portfolios)





best_weight=max_sharpe_idx


# 1. Grab the 1D weight vector at index `best_weight`
optimal_weights = return_weights[best_weight]

# 2. Map each asset i to its corresponding weight optimal_weights[i]
weights_dict = {
    f"Asset {i + 1}": [f"{optimal_weights[i] * 100:.2f}%"]
    for i in range(assets)
}

# 3. Create a DataFrame
weights_df = pd.DataFrame(weights_dict, index=["Optimal Allocation"])

# 4. Render directly in Streamlit
st.subheader("Optimal Portfolio Weights")
st.write("The optimal weights for each asset in the portfolio are as follows:")
st.dataframe(weights_df)

chart_df = (
    weights_df.T.iloc[:, 0]
    .str.rstrip("%")
    .astype(float)
    .rename("Weight (%)")
)
st.bar_chart(chart_df)
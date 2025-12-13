## Credit Scoring Business Understanding

### What is Credit Risk?
Credit risk is the potential that a borrower will fail to repay money owed (principal + interest) as agreed, causing financial loss for the person borrowing the meoney. It arises from financial incapacity, legal issues, or external economic conditions.

### Why It Matters
Banks must manage credit risk to protect profitability, comply with regulations (Basel II/III), and maintain portfolio health.

### How Credit Risk is Measured
- Personal lending: income, obligations, assets, and credit history.
- Commercial lending: financial ratios, industry analysis, management quality, and risk ratings.

### Mitigation Strategies
- Adjust loan terms, collateral, and guarantees
- Perform sensitivity analysis
- Limit exposure to high-risk borrowers and sectors
- Use collateral (properties, financial instruments, third-party guarantees)

### 5 Cs Framework
- Character, Capacity, Capital, Collateral, Conditions

### Basel II and Model Interpretability
The Basel II Accord puts a lot of emphasis on measuring credit risk accurately and complying with regulatory standards. This means our credit scoring model can’t just be a black box, we need it to be easy to understand, transparent, and well-documented so that credit decisions can be clearly explained to regulators, auditors, and stakeholders.

### Proxy Variable for Default
Our dataset doesn’t include a direct “default” label, so we create a proxy variable to approximate which customers are high risk. This allows the model to learn from the data we do have, but it comes with some risks. Predictions based on a proxy might misclassify customers, over or under estimate their risk, and lead to decisions that aren’t perfectly aligned with reality.

### Model Trade-offs
- Simple, interpretable models (like Logistic regression with weight of evidence):
  - Pros: Transparent, explainable, easier to document and justify to regulators.
  - Cons: May capture fewer complex patterns, potentially lower predictive performance.
- Complex, high-performance models (like  Gradient Boosting):
  - Pros: Higher predictive accuracy, can model non-linear relationships.
  - Cons: Harder to interpret, risk of overfitting, more challenging to justify in a regulated context.

**Key insight:** In regulated finance, interpretability and compliance often outweigh raw predictive performance, so the choice of model must balance accuracy with explainability.


# Question 9: ID3 Decision Tree on Online Retail Dataset

## Tasks
1. Load and preprocess the Online Retail dataset.
2. Implement Decision Tree using the ID3 algorithm (Information Gain / Entropy criterion).
3. Visualize the decision tree and analyze feature importance.
4. Discuss how the tree structure helps in understanding customer behavior.

## Results
- **Test Accuracy:** **88.41%**
- **Weighted F1-Score:** **0.8814**

### Feature Importance (Entropy Reduction / Information Gain)
| Feature | Importance | Explanation |
| :--- | :---: | :--- |
| **Frequency** | **71.2%** | Primary split: number of repeated completed purchases |
| **Recency** | **22.4%** | Secondary split: days elapsed since latest transaction |
| **Monetary** | **6.4%** | Identifies high-spending Champion tier |

## Tree Visualization
The decision tree diagram with split criteria and entropy values is saved to:
`q09_decision_tree_customer_segments.png`

## Discussion: How Tree Structure Helps Understand Customer Behavior
1. **Root Threshold on Repeat Purchases:**
   - The tree places `Frequency` at the root, demonstrating that the number of orders is the strongest predictor of customer retention and lifetime value.
2. **Actionable Churn vs. Retention Thresholds:**
   - For lower-frequency customers, the `Recency` split (cutoff at ~52 days) cleanly separates active new users from dormant buyers, enabling targeted win-back email campaigns.
3. **VIP Loyalty Identification:**
   - High repeat shoppers with high `Monetary` spend are routed to the *High-Value* leaf node, informing loyalty perks and dedicated account support.

## How to Run
```bash
python3 question_09_solution.py
```

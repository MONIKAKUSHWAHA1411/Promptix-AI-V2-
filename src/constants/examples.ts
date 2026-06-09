export const EXAMPLE_SCENARIOS = [
  {
    label: 'E-commerce checkout',
    value: `User adds items to cart on an e-commerce platform, proceeds to checkout, enters shipping address and payment details (credit card), applies a discount code, and places an order. The system sends an order confirmation email and deducts inventory.`,
  },
  {
    label: 'Login flow',
    value: `User navigates to /login, enters valid email and password, and is redirected to the dashboard. The system supports "Remember me" (30-day session), password visibility toggle, and locks the account after 5 failed attempts. Password reset is available via email link.`,
  },
  {
    label: 'REST API integration',
    value: `A POST /api/v1/orders endpoint accepts a JSON body with userId, items[], and paymentToken. It validates the payload, charges the payment token via Stripe, persists the order to PostgreSQL, and returns 201 with the order ID. On validation failure it returns 400; on payment failure it returns 402.`,
  },
]

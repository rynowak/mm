---
name: playwright
description: |
  Playwright end-to-end testing skill. Covers test tiers (component, integration,
  e2e), page object patterns, selector strategies, waiting/assertions, CI
  configuration, and debugging. Use when user asks to "write playwright tests",
  "add e2e tests", "test the UI", "run playwright", "add browser tests", or
  works on UI test infrastructure.
file_dependencies:
  - path: docs/ui-test-context.md
    description: "Playwright version, test conventions, auth strategy, config location"
    template: templates/ui-test-context.md.tmpl
---

# Playwright Testing

## Overview

Write and maintain Playwright end-to-end tests. Follow the test tier model,
use stable selectors, avoid flaky patterns, and structure tests for
maintainability and CI reliability.

## When to Use

- Adding or reviewing browser-based tests
- Testing UI components, flows, or integrations
- Setting up Playwright infrastructure for a project
- Debugging flaky e2e tests

## Dependencies

Read repo-specific UI context before writing tests:
- `AGENTS.md` or `playwright.config.ts` for existing test conventions
- Existing test files for patterns already in use
- `package.json` for Playwright version and scripts

## Test Tiers

| Tier | Scope | Backend | Speed |
|------|-------|---------|-------|
| **Component** | Isolated UI rendering and interaction | Mocked (MSW, route interception) | Fast |
| **Integration** | UI + mock API server | Local mock server | Medium |
| **E2E** | Full stack with real backend | Dev/staging server | Slow |

Start with the tier appropriate to what you're testing. Prefer lower tiers
for logic validation; use E2E only for critical user flows.

## Selector Strategy

**Priority order (most stable to least stable):**

1. `getByRole()` — accessible roles (`button`, `heading`, `textbox`)
2. `getByTestId()` — explicit `data-testid` attributes
3. `getByText()` / `getByLabel()` — visible text and form labels
4. `getByPlaceholder()` — form placeholder text
5. CSS/XPath — last resort only

**Never use:**
- Auto-generated class names (CSS modules, Tailwind arbitrary values)
- DOM structure paths (`div > div > span:nth-child(3)`)
- Indexes without semantic context

## Waiting and Assertions

**Playwright auto-waits.** Do not add explicit waits unless truly needed.

```typescript
// GOOD — auto-waits for element
await expect(page.getByRole('heading')).toHaveText('Dashboard');

// GOOD — wait for network idle after navigation
await page.goto('/dashboard');
await page.waitForLoadState('networkidle');

// BAD — arbitrary timeout
await page.waitForTimeout(2000);
```

**Assertions to prefer:**
- `toBeVisible()` — element is rendered and visible
- `toHaveText()` / `toContainText()` — text content
- `toHaveCount()` — number of matching elements
- `toHaveURL()` — navigation verification
- `toBeEnabled()` / `toBeDisabled()` — interactive state

## Page Object Pattern

For flows spanning multiple interactions, use page objects:

```typescript
class LoginPage {
  constructor(private page: Page) {}

  async login(email: string, password: string) {
    await this.page.getByLabel('Email').fill(email);
    await this.page.getByLabel('Password').fill(password);
    await this.page.getByRole('button', { name: 'Sign in' }).click();
  }

  async expectError(message: string) {
    await expect(this.page.getByRole('alert')).toContainText(message);
  }
}
```

Use page objects when:
- The same page interactions appear in multiple tests
- A flow has 3+ steps worth encapsulating
- The page structure is complex enough to benefit from abstraction

Do not over-abstract — a simple test with inline selectors is better than
a page object used once.

## Network Interception

Use `page.route()` for mocking API responses in component/integration tiers:

```typescript
await page.route('**/api/users', async route => {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify([{ id: 1, name: 'Test User' }]),
  });
});
```

Use for:
- Isolating UI from backend state
- Testing error states (4xx, 5xx, timeouts)
- Controlling data for deterministic assertions

## Authentication

For tests requiring auth:
- **Component tier**: Mock auth state entirely (inject tokens, mock auth providers)
- **Integration tier**: Use a mock auth endpoint that returns test tokens
- **E2E tier**: Use `storageState` to reuse authenticated sessions:

```typescript
// Save auth state once
const context = await browser.newContext();
// ... perform login ...
await context.storageState({ path: 'auth.json' });

// Reuse in tests
const context = await browser.newContext({ storageState: 'auth.json' });
```

## CI Configuration

Key practices for reliable CI runs:

- **Retries**: Set `retries: 2` in CI (0 locally) to handle infrastructure flakes
- **Workers**: Match to CI runner cores (typically 2-4)
- **Screenshots/traces**: Capture on failure only (`'only-on-failure'`)
- **Base URL**: Configure via env var, not hardcoded
- **Browser**: Default to Chromium only in CI unless cross-browser is required
- **Timeouts**: Set explicit `timeout` and `expect.timeout` in config

```typescript
// playwright.config.ts
export default defineConfig({
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'only-on-failure',
    screenshot: 'only-on-failure',
  },
});
```

## Debugging

- `npx playwright test --ui` — interactive test runner
- `npx playwright test --debug` — step through with inspector
- `npx playwright show-trace trace.zip` — view failure traces
- `page.pause()` — pause execution at a specific point
- `PWDEBUG=1` — enable headed mode with inspector

## Anti-Patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| `waitForTimeout(N)` | Flaky, slow | Use auto-wait assertions |
| Testing implementation details | Brittle | Test user-visible behavior |
| One massive test | Hard to debug, slow | Split into focused scenarios |
| Shared mutable state between tests | Order-dependent | Isolate with `beforeEach` |
| Asserting exact snapshot for dynamic content | False failures | Assert structure, not content |
| Login in every test | Slow | Use `storageState` |

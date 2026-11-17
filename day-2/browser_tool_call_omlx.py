"""The oMLX twin of browser_tool_call.py: the same 4B model with a real browser as its tool, but Qwen3.5-4B is
served by oMLX's OpenAI-compatible /v1/chat/completions instead of Ollama. It has two tools that drive YOUR logged-in
Ego Lite browser and hand back Markdown:

  read_open_profile()   the LinkedIn profile you have open in Ego Lite right now: it clicks every "… more",
                        follows "Show all experiences" (and education, skills…), and saves everything to one file
                        named after the profile (linkedin.com/in/jentaylor2 -> jentaylor2.md), with the photo
  read_page(url)        any page, opened in a fresh tab

    python browser_tool_call_omlx.py                  # "Read the LinkedIn profile I have open, all of it, as Markdown"
    python browser_tool_call_omlx.py "Read https://en.wikipedia.org/wiki/Berlin and list its districts"

Before you run it:
  1. Install Ego Lite (https://lite.ego.app), a Chromium browser built for people AND agents, log in to LinkedIn
     yourself and open a profile. The model never sees your password: it borrows the browser you logged in to.
  2. Run oMLX on :8000 with mlx-community/Qwen3.5-4B-MLX-4bit in the Hugging Face cache, and put OMLX_API_KEY in
     the environment or the repo-root .env. Needs `pip install requests python-dotenv`.

Each profile becomes three files in day-2/output/, named after the profile address (linkedin.com/in/jentaylor2):
jentaylor2.json (Proxycurl's profile format, as in the 2024 archive), jentaylor2.md (laid out from the JSON by profile_markdown.py) and jentaylor2.jpg.
read_page() writes output/page-omlx.md, and the model's own answer goes to output/answer-omlx.md (-omlx so they
don't overwrite the Ollama run's files).
Only read profiles you are allowed to read: LinkedIn's terms forbid automated scraping, and a profile is someone's
personal data. Ask first, keep the file to yourself, and delete it when you're done.

The same idea, three ways (see the slide):
  - here:        a Python tool that pipes a script to `ego-browser nodejs`
  - Claude Code: the ego-browser skill (~/.claude/skills/ego-browser) - Claude writes these browser scripts itself
  - steel.dev:   a self-hosted browser API; connect with Playwright over CDP (ws://localhost:3000) instead
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
HEADERS = {"Authorization": f"Bearer {api_key}"}

OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mlx-community--Qwen3.5-4B-MLX-4bit"   # oMLX names Hugging Face cache models "org--repo"
THINKING = False          # a 4B overthinks tool calls; off is faster and picks the tool just as well
OUT = Path(__file__).parent / "output"
MAX_CHARS = 12000         # what we hand back to the model; a 4B's context fills fast (the file gets everything)

# ---- browser-side helpers, shared by both tools (they run inside Ego Lite) -------------------------------------
HELPERS = r"""
const fs = await import("node:fs/promises");    // long output can land on stderr, so results go to a file
// Every pause is a little random (0.7 to 1.5 times the nominal time), like a person reading, not a metronome.
const sleep = (ms) => new Promise((r) => setTimeout(r, Math.round(ms * (0.7 + Math.random() * 0.8))));

// Scroll the page (and LinkedIn's inner scroll box) to the bottom, slowly, until nothing new loads: LinkedIn
// fetches sections and long lists (all 61 skills, say) only as they scroll into view.
async function scrollAll(page) {
  let last = -1, still = 0;
  for (let i = 0; i < 90 && still < 5; i++) {               // stop after 5 rounds with nothing new
    const state = await page.evaluate(() => {
      const boxes = [document.scrollingElement, ...document.querySelectorAll("main, main *")]
        .filter((e) => e && e.scrollHeight > e.clientHeight + 50 && (e === document.scrollingElement || /auto|scroll/.test(getComputedStyle(e).overflowY)));
      let moved = false;
      const step = 600 + Math.floor(Math.random() * 400);           // 600-1000 px, a different step each time
      for (const b of boxes) { const before = b.scrollTop; b.scrollTop += step; if (b.scrollTop !== before) moved = true; }
      return { moved, len: (document.querySelector("main") || document.body).innerText.length };
    });
    await sleep(900);
    still = (!state.moved && state.len === last) ? still + 1 : 0;
    last = state.len;
  }
}

// Long lists ("Show all" pages) load in batches: scroll, press "Show more results" until it is gone.
async function loadAll(page) {
  for (let i = 0; i < 40; i++) {
    await scrollAll(page);
    const more = await page.evaluate(() => {
      const b = [...document.querySelectorAll("main button")].find((e) => /^show more( results)?$/i.test((e.innerText || "").trim()));
      if (!b) return false;
      b.click();
      return true;
    });
    if (!more) break;
    await sleep(1500);
  }
}

// Click every "… more" / "see more" expander in the profile sections, one at a time. Never the "More" menu
// buttons (they open menus such as Save to PDF) and never anything in Activity: a post's "… more" opens the post.
async function expandAll(page) {
  const home = await page.url();
  let total = 0;
  for (let round = 0; round < 5; round++) {
    const n = await page.evaluate(() => {
      const start = location.href;
      const text = (b) => (b.innerText || "").trim();
      const isExpander = (b) => (/^(…|\.\.\.)\s*(see )?more$/i.test(text(b)) || /^see more$/i.test(text(b))
                                 || /inline-show-more-text__button/.test(b.className))
                                && b.getAttribute("aria-expanded") !== "true";
      const inActivity = (b) => /activity|posts/i.test(b.closest("section")?.querySelector("h2")?.innerText || "")
                                || b.closest("a, .feed-shared-update-v2, [data-urn*='activity']");
      let clicked = 0;
      for (const b of [...document.querySelectorAll("main button")].filter((b) => isExpander(b) && !inActivity(b))) {
        b.click();
        clicked++;
        if (location.href !== start) return -1;      // it navigated: stop, the caller goes back
      }
      return clicked;
    });
    if (n === -1 || (await page.url()) !== home) {
      await page.goto(home, { timeout: 30000 });
      await page.waitForLoadState("load");
      await sleep(1500);
      break;
    }
    total += n;
    if (!n) break;
    await sleep(500);
  }
  return total;
}

// The page as structured data: for each section, its entries (one job, school, certificate or recommendation each).
// LinkedIn's sections are runs of text, not lists, so an entry is found from its date ("2014 – Present",
// "Issued Sep 2019", "November 13, 2026"): the largest block around one date that holds no other date.
async function extract(page) {
  return page.evaluate(() => {
    const DATE = /((19|20)\d{2}.*[–-].*((19|20)\d{2}|Present))|^Issued |^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}|^(19|20)\d{2}$/;
    const NOISE = /^(Follow|Following|Message|More|Show all|Contact info|·|•|-|Request services|Connect|Pending|Show credential|See all|Show more|Show less|Www|Endorse)$/;
    const lines = (el) => {
      const out = [];
      for (const l of (el.innerText || "").split("\n").map((s) => s.trim()).filter((s) => s && !NOISE.test(s)))
        if (out[out.length - 1] !== l) out.push(l);
      return out;
    };
    const visible = (e) => { const st = getComputedStyle(e); return st.display !== "none" && st.visibility !== "hidden"; };
    return [...document.querySelectorAll("main section")].filter(visible).map((s) => {
      const title = (s.querySelector("h2")?.innerText || "").split("\n")[0].trim();
      const anchors = [...s.querySelectorAll("*")].filter((e) => !e.children.length && DATE.test((e.innerText || "").trim()));
      const dates = (el) => new Set(anchors.filter((a) => el.contains(a)).map((a) => a.innerText.trim())).size;
      const entries = [];
      for (const a of anchors) {
        let el = a;
        while (el.parentElement && el.parentElement !== s && dates(el.parentElement) === 1) el = el.parentElement;
        if (!entries.some((e) => e === el || e.contains(el))) entries.push(el);
      }
      const posts = [...s.querySelectorAll("li")].filter((li) => !li.parentElement.closest("li")).map(lines);
      return { title, lines: lines(s), entries: entries.map(lines), posts };
    });
  });
}

// The page as simple Markdown: headings, list items, paragraphs. Skips menus, buttons and "people you may know".
async function toMarkdown(page) {
  return page.evaluate(() => {
    const SKIP = /people you may know|you might like|more profiles for you|people also viewed|pages for you/i;
    const root = document.querySelector("main") || document.body;
    const out = [];
    const walk = (el) => {
      if (el.nodeType === 3) { const t = el.textContent.replace(/\s+/g, " "); if (t.trim()) out.push(t); return; }
      if (el.nodeType !== 1) return;
      const tag = el.tagName.toLowerCase(), style = getComputedStyle(el);
      if (["script", "style", "noscript", "svg", "button", "nav", "footer", "img"].includes(tag)) return;
      if (style.display === "none" || style.visibility === "hidden" || el.getAttribute("aria-hidden") === "true") return;
      if (tag === "section" && SKIP.test(el.querySelector("h2")?.innerText || "")) return;
      if (/^h[1-6]$/.test(tag)) { out.push("\n\n" + "#".repeat(Math.min(+tag[1] + 1, 6)) + " " + el.innerText.trim().split("\n")[0] + "\n\n"); return; }
      if (tag === "li") out.push("\n- ");
      if (["p", "div", "section", "article", "br"].includes(tag)) out.push("\n");
      for (const c of el.childNodes) walk(c);
    };
    walk(root);
    return out.join("").replace(/[ \t]+\n/g, "\n").replace(/\n- \n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  });
}
"""

# read_page(url): a fresh agent tab, closed again afterwards.
PAGE_SCRIPT = HELPERS + r"""
const task = await taskSpace("read-page " + Date.now());
const page = task.page("p1");
await page.goto(%(url)s, { timeout: 30000 });
await page.waitForLoadState("load");
await sleep(2500);                                     // let a JavaScript-heavy page finish drawing
const md = "# " + (await page.title()) + "\n\n" + (await page.url()) + "\n\n" + (await toMarkdown(page));
await task.finish({ keep: [] });
await fs.writeFile(%(out)s, md);
"""

# read_open_profile(): read the LinkedIn profile in the user's own tab (only that tab has every section loaded;
# LinkedIn leaves them as grey placeholders in a fresh agent tab). In the user's tab we only scroll and click the
# "… more" text expanders. The full "Show all" lists are then read in a separate agent tab, closed afterwards.
PROFILE_SCRIPT = HELPERS + r"""
// Which profile: the address we were given, or the LinkedIn profile open in the user's Ego Lite (only looked at).
let profileUrl = %(url)s;
if (!profileUrl) {
  const entry = (await listTaskSpaces()).find((s) => s.ownership !== "agent" && (s.recentTabTitles || []).some((t) => /LinkedIn/.test(t)));
  if (!entry) throw new Error("No LinkedIn tab is open in Ego Lite: open the profile there first.");
  const mine = entry.ownership === "user" ? await claimTaskSpace(entry.id) : await takeOverTaskSpace(entry.id);
  const tab = (await mine.tabs()).find((t) => /linkedin\.com\/in\//.test(t.url));
  await mine.finish({ keep: [] });                      // hands the space straight back; the user's tab stays as it was
  if (!tab) throw new Error("The open LinkedIn tab is not a profile (linkedin.com/in/...).");
  profileUrl = tab.url;
}
profileUrl = profileUrl.split("?")[0].replace(/\/?$/, "/");
const slug = profileUrl.split("/in/")[1].split("/")[0];

// Everything else happens in the agent's own tab (same browser, same login), closed at the end.
const task = await taskSpace("read-profile " + Date.now());
const page = task.page("p1");
await page.goto(profileUrl, { timeout: 30000 });
await page.waitForLoadState("load");
await sleep(3000);
await scrollAll(page);
for (let tries = 0; tries < 3; tries++) {             // LinkedIn sometimes leaves sections as grey placeholders
  const loaded = await page.evaluate(() => [...document.querySelectorAll("main section h2")].some((h) => /^(Experience|About)/.test(h.innerText)));
  if (loaded) break;
  await page.evaluate(() => { document.querySelector("main")?.scrollTo(0, 0); window.scrollTo(0, 0); });
  await sleep(3000);
  await scrollAll(page);
}
let clicks = await expandAll(page);
const name = (await page.title()).replace(/\s*\|\s*LinkedIn.*$/, "").replace(/^\(\d+\)\s*/, "");
// The profile photo: the first picture in the profile with a srcset (the top card; the cover photo is skipped),
// at its largest size (LinkedIn offers 100, 200, 400 and 800 px). Fallback: a picture whose alt names the person.
const photoSrc = await page.evaluate((name) => {
  const sized = [...document.querySelectorAll("main img")].filter((i) => i.srcset && !/cover|background|banner/i.test(i.alt));
  if (sized.length) {
    return sized[0].srcset.split(",").map((s) => s.trim().split(/\s+/)).map(([u, w]) => ({ u, w: parseInt(w) || 0 }))
      .sort((a, b) => b.w - a.w)[0].u;
  }
  const named = [...document.querySelectorAll("main img")].filter((i) => (i.alt || "").includes(name));
  return named[0]?.src || null;
}, name);
let photo = null;
if (photoSrc) {
  try { await page.fetch(photoSrc, { saveAs: %(dir)s + "/" + slug + ".jpg" }); photo = slug + ".jpg"; } catch (e) { photo = null; }
}
const coverUrl = await page.evaluate(() => [...document.querySelectorAll("main img")].find((i) => /cover|background/i.test(i.alt))?.src || null);
const data = { name, slug, profileUrl, photo, photoUrl: photoSrc, coverUrl, clicks: 0, sections: await extract(page), details: {} };

// Contact info: open the pop-up the way a person does, read it, close it. (1st-degree connections only show
// what they chose to share: email, phone, websites, birthday, "connected since".)
data.contact = null;
try {
  await page.click("text=Contact info", { label: "open contact info" });
  await sleep(2500);
  data.contact = await page.evaluate(() => {
    const box = document.querySelector("[role=dialog], dialog, [aria-modal=true]")
      || [...document.querySelectorAll("div, section")].filter((d) => /^Contact info/.test((d.innerText || "").trim()) && /Connected since|Email|Website|Phone/.test(d.innerText)).sort((a, b) => a.innerText.length - b.innerText.length)[0];
    if (!box) return null;
    const out = [];
    for (const l of box.innerText.split("\n").map((s) => s.trim()).filter(Boolean))
      if (out[out.length - 1] !== l) out.push(l);
    return out;
  });
  await page.keyboard.press("Escape");
  await sleep(1000);
  if (!/\/in\/[^/]+\/?$/.test(new URL(await page.url()).pathname)) { await page.goto(profileUrl, { timeout: 30000 }); await sleep(2000); }
} catch (e) { /* no Contact info link (not a connection), or it didn't open */ }

// Interests: the section opens on one tab (often people); click Companies and Groups like a person would.
data.interests = {};
for (const tabName of ["Companies", "Groups"]) {
  const clicked = await page.evaluate((tabName) => {
    const sec = [...document.querySelectorAll("main section")].find((s) => /^Interests/.test(s.querySelector("h2")?.innerText || ""));
    const tab = sec && [...sec.querySelectorAll("*")].find((e) => !e.children.length && (e.innerText || e.textContent || "").trim() === tabName);
    if (!tab) return false;
    (tab.closest("button, [role=tab], [role=button], a, label") || tab).click();
    return true;
  }, tabName);
  if (!clicked) continue;
  await sleep(1800);
  data.interests[tabName] = await page.evaluate(() => {
    const sec = [...document.querySelectorAll("main section")].find((s) => /^Interests/.test(s.querySelector("h2")?.innerText || ""));
    const out = [];
    for (const l of (sec?.innerText || "").split("\n").map((s) => s.trim()).filter(Boolean)) if (out[out.length - 1] !== l) out.push(l);
    return out;
  });
}

// "Show all …" pages hold the full lists: every job, skill, recommendation, award…
const details = await page.evaluate((slug) => [...new Set([...document.querySelectorAll("main a[href*='/details/']")]
  .map((a) => a.href.split("?")[0])
  .filter((h) => new RegExp("/in/" + slug + "/details/(experience|education|skills|recommendations|courses|certifications|projects|languages|honors|publications|volunteering-experiences|organizations)/$").test(h)))], slug);
for (const url of details) {
  await sleep(2000);                                    // a pause before each new page, as a reader would
  await page.goto(url, { timeout: 30000 });
  await page.waitForLoadState("load");
  await sleep(3500);
  await loadAll(page);
  clicks += await expandAll(page);
  await loadAll(page);                                  // expanding can load more below
  data.details[url.split("/details/")[1].replace(/\/$/, "")] = await extract(page);
}
await task.finish({ keep: [] });                        // close the agent tab
data.clicks = clicks;
await fs.writeFile(%(out)s, JSON.stringify(data));   // raw page data; profile_markdown.py turns it into Proxycurl JSON
"""


def run_ego(script, out_file):
    out_file.unlink(missing_ok=True)
    run = subprocess.run(["ego-browser", "nodejs"], input=script, capture_output=True, text=True, timeout=300)
    if run.returncode != 0 or not out_file.exists():
        return None, (run.stderr or run.stdout).strip()[-400:]
    return out_file.read_text().strip(), ""


def read_page(url):
    """Tool: open any page in the user's own logged-in browser and return it as Markdown."""
    print(f"  🌐 opening {url} in Ego Lite …")
    OUT.mkdir(exist_ok=True)
    out = (OUT / "page-omlx.md").resolve()
    md, err = run_ego(PAGE_SCRIPT % {"url": json.dumps(url), "out": json.dumps(str(out))}, out)
    if md is None:
        return f"Could not read {url}: {err}"
    if "/authwall" in md[:800] or "/login" in md[:800]:
        return f"{url} needs a login. Ask the user to log in to the site in Ego Lite, then try again."
    print(f"  📄 {len(md):,} characters, saved to {out}")
    return md[:MAX_CHARS]


def read_open_profile(url=""):
    """Tool: a LinkedIn profile (the one open in the user's browser, or `url`), every "… more" expanded.
    Saves output/<profile>.json (clean data), output/<profile>.md (laid out from the JSON) and output/<profile>.jpg."""
    from profile_markdown import to_proxycurl, render
    print("  🌐 reading the LinkedIn profile open in Ego Lite (expanding every '… more') …")
    OUT.mkdir(exist_ok=True)
    tmp = (OUT / "profile.raw.json").resolve()
    raw, err = run_ego(PROFILE_SCRIPT % {"url": json.dumps(url or ""), "out": json.dumps(str(tmp)),
                                         "dir": json.dumps(str(OUT.resolve()))}, tmp)
    if raw is None:
        return f"Could not read the open profile: {err}"
    tmp.unlink()
    raw = json.loads(raw)
    profile = to_proxycurl(raw)                         # page text -> Proxycurl's schema (+ our extras in "extra")
    stem = OUT / re.sub(r"[^A-Za-z0-9_-]+", "", raw["slug"])            # named after the profile: jentaylor2
    stem.with_suffix(".json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    md = render(profile)                                # JSON -> Markdown
    stem.with_suffix(".md").write_text(md)
    photo = profile["extra"]["profile_pic_file"]
    print(f"  📄 {profile['full_name']}: clicked {raw['clicks']} '… more', followed {len(raw['details'])} 'Show all' page(s); "
          f"{len(profile['experiences'])} roles, {len(profile['skills'])} skills, "
          f"{len(profile['recommendations'])} recommendations")
    print(f"     saved {stem.name}.json, {stem.name}.md" + (f" and {photo}" if photo else " (no photo found)"))
    return md[:MAX_CHARS]


TOOLS = {"read_page": read_page, "read_open_profile": read_open_profile}
tools = [
    {"type": "function", "function": {
        "name": "read_open_profile",
        "description": "Read the LinkedIn profile the user has open in their browser right now, with every "
                       "'… more' expanded and the full experience/education lists, and save it all to a Markdown file. "
                       "Use it whenever the user mentions the profile they have open.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Optional linkedin.com/in/... address; leave empty for the open profile"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "read_page",
        "description": "Open a web page in the user's own logged-in browser and return its content as Markdown.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string", "description": "The full https:// address"}},
                       "required": ["url"]}}},
]


def chat(messages):
    r = requests.post(OMLX_URL, headers=HEADERS,
                      json={"model": MODEL, "messages": messages, "tools": tools, "stream": False,
                            "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": THINKING}}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "Read the LinkedIn profile I have open in my browser and write it all out as clean Markdown: " \
        "name, headline, about, every job with its description, education and skills."
    print(f"USER: {question}\n")
    messages = [{"role": "system", "content": "You can use a web browser through your tools. Answer in Markdown."},
                {"role": "user", "content": question}]

    for _ in range(4):                                  # the agent loop: call tools until the model answers
        msg = chat(messages)
        messages.append(msg)
        if not msg.get("tool_calls"):
            break
        for call in msg["tool_calls"]:
            fn, args = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")   # a JSON string
            print(f"MODEL calls {fn}({args})")
            result = TOOLS[fn](**args) if fn in TOOLS else f"unknown tool {fn}"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    OUT.mkdir(exist_ok=True)
    (OUT / "answer-omlx.md").write_text(msg["content"] or "")
    print(f"\nMODEL:\n{msg['content']}\n\n(saved to {OUT / 'answer-omlx.md'})")

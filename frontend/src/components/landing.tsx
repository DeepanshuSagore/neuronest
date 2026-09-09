"use client";

import {
  animate,
  spring,
  createTimeline,
  onScroll,
  splitText,
  stagger,
  utils,
  type AnimationParams,
  type Scope,
  type ScrollObserver,
} from "animejs";
import clsx from "clsx";
import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import { twMerge } from "tailwind-merge";

import { METRICS_ARE_PLACEHOLDER, content, type Citation, type Metric } from "@/lib/content";
import { prefersReducedMotion, useAnimeScope } from "@/lib/motion";

/**
 * Press — NeuroNest as a two-colour print job.
 *
 * The premise: this project publishes honest measurements, including the
 * negative ones, and a poster is the format that states a thing bluntly and
 * refuses to hedge. So the page is set the way a broadside is set — heavy
 * black rules, flat fills, oversized display type, blocks that overlap and sit
 * a few pixels out of register the way a second plate does when the paper
 * shifts on the drum.
 *
 * The discipline is crude materials, exact craft. Nothing here is rotated for
 * fun, nothing is randomly placed, and every block lands on the same twelve
 * columns. The surface is loud; the setting is not careless.
 */

/* ══════════════════════════════════════════════════════════════════════════
   TOKENS
   ══════════════════════════════════════════════════════════════════════════

   Scoped to `.press` rather than `:root`. Four other directions mount at
   neighbouring URLs inside the same document shell, and a global custom
   property would leak straight into them.

   Utility classes live here too — border widths and hard shadows are
   expressed as `var(--pr-rule)` / `var(--pr-drop)`, which Tailwind's arbitrary
   syntax handles awkwardly for lengths. Declaring them once as `.pr-*` keeps
   every weight and offset in this block and out of the markup. */

const PRESS_CSS = `
.press {
  /* ── Stock ───────────────────────────────────────────────────────────
     Warm, not white. --pr-stock is the ground the page prints on and every
     ratio below is measured against it; --pr-sheet is the lighter sheet a
     block is printed on, where each ink scores higher still. */
  --pr-stock:  #f2ebdc;
  --pr-sheet:  #fbf7ee;

  /* ── Inks ────────────────────────────────────────────────────────────
     Three blacks, all AA at the size they are set. */
  --pr-ink:    #151310; /* 15.6:1 — display, headings, body, every rule */
  --pr-ink-2:  #57524a; /*  6.5:1 — decks and detail copy */
  --pr-ink-3:  #6b665c; /*  4.8:1 — captions and small labels, still AA */

  /* The flag. 2.6:1 on stock, 2.9:1 on sheet — under the 3:1 large-text
     floor, so it is a FILL AND BORDER COLOUR and never carries a glyph that
     means anything. Black on it reads 6.0:1, which is why every orange block
     on this page is set in --pr-ink rather than in stock. */
  --pr-flag:   #ff5c35;

  /* The same hue burnt down until it is legal as type: 4.8:1 on stock,
     5.3:1 on sheet. Anything that must be orange AND readable is this. */
  --pr-flag-ink: #c0330d;

  /* What is set ON a solid flag block. Ink here only because ink happens to be
     dark; a palette that lightens ink has to restate this or every key and
     badge loses its type. */
  --pr-on-flag: var(--pr-ink);

  /* The blueline. A press pulls a blue proof before the real run, which is
     exactly the status of every figure on this page. 7.1:1 on stock, 7.9:1
     on sheet, and stock on a solid blueline block is 7.1:1 back. */
  --pr-proof:  #2438c9;

  /* ── Rules and registration ──────────────────────────────────────────
     Mobile-first: thinner rules and shorter drops on a small sheet, stepped
     up at sm. Reading these back out of the computed style is how the motion
     code learns where a ghost plate comes to rest, so the offsets exist in
     exactly one place. */
  --pr-rule:    2px;
  --pr-rule-2:  3px;
  --pr-drop-sm: 3px;
  --pr-drop:    5px;
  --pr-drop-lg: 7px;

  /* Halftone density and the one tilt on the page, so a stamp cannot drift. */
  --pr-screen: 0.14;
  --pr-tilt:  -2.5deg;

  /* CSS-driven durations. The JS timeline keeps its own constants — anime
     needs numbers before first paint, and reading computed styles per
     animation would be a layout read for no benefit. */
  --pr-press:  90ms;
  --pr-settle: 220ms;

  /* ── Type ────────────────────────────────────────────────────────────
     A poster is one enormous size and one small one; the middle of the scale
     is deliberately thin so the jump reads as intended rather than as drift. */
  --pr-mega:    clamp(2.75rem, 12vw, 9.5rem);
  --pr-display: clamp(2rem, 6.4vw, 4.5rem);
  --pr-lead:    clamp(1.5rem, 3.4vw, 2.5rem);
  --pr-ask:     clamp(1.25rem, 2.6vw, 1.875rem);
  --pr-stage:   clamp(1.375rem, 2.4vw, 1.875rem);
  --pr-numeral: clamp(3.25rem, 9vw, 6.5rem);
  --pr-figure:  clamp(1.5rem, 3.4vw, 2.75rem);

  background-color: var(--pr-stock);
  color: var(--pr-ink);
  -webkit-font-smoothing: antialiased;
}

@media (min-width: 640px) {
  .press {
    --pr-rule:    3px;
    --pr-rule-2:  4px;
    --pr-drop-sm: 5px;
    --pr-drop:    8px;
    --pr-drop-lg: 12px;
  }
}

.press ::selection {
  background-color: var(--pr-ink);
  color: var(--pr-stock);
}

/* ── Focus ──────────────────────────────────────────────────────────────
   A blueline ring inside a stock-coloured halo. The halo is what makes one
   ring legal everywhere: blue on stock is 7.1:1 and blue on the flag is
   3.2:1, but blue on a solid black block would be 2.2:1 — so the halo puts
   a known 15.6:1 field behind the ring no matter what it lands on. */
.press :focus-visible {
  outline: var(--pr-rule-2) solid var(--pr-proof);
  outline-offset: var(--pr-rule);
  box-shadow: 0 0 0 calc(var(--pr-rule) + var(--pr-rule-2)) var(--pr-stock);
  border-radius: 0;
}

/* Ghost plates sit at inset-0 and are pushed out by --pr-drop, so the rightmost
   one overhangs the page and opens a horizontal scrollbar on a phone. Clipped
   rather than hidden: 'hidden' would make this a scroll container and kill the
   sticky index inside it. */
.press { overflow-x: clip; }

/* ── Rules ──────────────────────────────────────────────────────────────
   Every edge on this page is one of two weights, in ink. */
.press .pr-edge     { border: var(--pr-rule) solid var(--pr-ink); }
.press .pr-edge-2   { border: var(--pr-rule-2) solid var(--pr-ink); }
.press .pr-edge-t   { border-top: var(--pr-rule) solid var(--pr-ink); }
.press .pr-edge-t-2 { border-top: var(--pr-rule-2) solid var(--pr-ink); }
.press .pr-edge-b   { border-bottom: var(--pr-rule) solid var(--pr-ink); }
.press .pr-edge-b-2 { border-bottom: var(--pr-rule-2) solid var(--pr-ink); }
.press .pr-edge-r   { border-right: var(--pr-rule) solid var(--pr-ink); }
.press .pr-edge-l-2 { border-left: var(--pr-rule-2) solid var(--pr-ink); }
.press .pr-edge-proof { border: var(--pr-rule) solid var(--pr-proof); }

/* Rules *between* items rather than around them, so a last-child override is
   never needed and a stray trailing rule can never appear. */
.press .pr-ruled > * + *  { border-top: var(--pr-rule) dashed var(--pr-ink-3); }
.press .pr-strata > * + * { border-left: var(--pr-rule) solid var(--pr-ink); }

/* ── Ghost plates ───────────────────────────────────────────────────────
   The second pass, out of register. A flat rectangle sitting behind a block
   at a fixed offset — this is both the hard shadow and the misregistration,
   which is why there is no 'box-shadow' on a block anywhere.

   The offset is a transform rather than an inset so the motion code can take
   it over without touching layout. */
.press .pr-ghost    { transform: translate(var(--pr-drop), var(--pr-drop)); }
.press .pr-ghost-sm { transform: translate(var(--pr-drop-sm), var(--pr-drop-sm)); }
.press .pr-ghost-lg { transform: translate(var(--pr-drop-lg), var(--pr-drop-lg)); }

/* ── Keys ───────────────────────────────────────────────────────────────
   A button carries its shadow as a real box-shadow so that pressing it can
   collapse the shadow and move the cap the same distance — the cap goes down
   into the hole it was casting. Hover lifts it a third of the way back out.

   globals.css flattens transition-duration under 'reduce', which turns these
   into instant state changes rather than removing them. */
.press .pr-key {
  box-shadow: var(--pr-drop) var(--pr-drop) 0 0 var(--pr-ink);
  transition:
    transform var(--pr-press) linear,
    box-shadow var(--pr-press) linear,
    background-color var(--pr-settle) linear;
}

.press .pr-key:hover {
  transform: translate(calc(var(--pr-drop) * -0.35), calc(var(--pr-drop) * -0.35));
  box-shadow: calc(var(--pr-drop) * 1.35) calc(var(--pr-drop) * 1.35) 0 0 var(--pr-ink);
}

.press .pr-key:active {
  transform: translate(var(--pr-drop), var(--pr-drop));
  box-shadow: 0 0 0 0 var(--pr-ink);
}

/* The generic focus rule replaces box-shadow outright, which would delete a
   key's shadow the moment it was focused. Restated here with both. */
.press .pr-key:focus-visible {
  box-shadow:
    var(--pr-drop) var(--pr-drop) 0 0 var(--pr-ink),
    0 0 0 calc(var(--pr-rule) + var(--pr-rule-2)) var(--pr-stock);
}

/* ── Screen ─────────────────────────────────────────────────────────────
   The halftone is real geometry — a tiled SVG circle — rather than a
   gradient, because a dot screen is dots. This only sets its density. */
.press .pr-screen { opacity: var(--pr-screen); }

/* ── Stamp ──────────────────────────────────────────────────────────────
   One tilt, one value, used four times. A page where everything is rotated
   by a different amount reads as an accident. */
.press .pr-stamp { transform: rotate(var(--pr-tilt)); }

/* Space Grotesk's descenders drop well below a 0.82 line box, and the line
   masks splitText generates clip to their padding box — so the 'y' in "your"
   would be guillotined at rest. Padding grows the clip region and an equal
   negative margin hands the leading straight back, leaving the measured line
   positions untouched. */
.press .pr-headline * {
  padding-bottom: 0.2em;
  margin-bottom: -0.2em;
}

/* ── The entrance gate ──────────────────────────────────────────────────
   The pre-paint hidden state is the only part of the motion system that lives
   in CSS, and it has to: useAnimeScope runs in an effect, which is after the
   first paint, so hiding from JS would show one composed frame and then yank
   it away.

   Both conditions are load-bearing. Under 'reduce' the JS returns early, so
   nothing would ever restore an element; without scripting nothing would run
   at all. In either case this rule never applies, and the page renders
   complete, composed and readable. */
@media (prefers-reduced-motion: no-preference) and (scripting: enabled) {
  .press [data-reveal],
  .press [data-group] > * {
    opacity: 0;
  }
}
`;

/* ══════════════════════════════════════════════════════════════════════════
   LAYOUT AND TYPE PRIMITIVES
   ══════════════════════════════════════════════════════════════════════════ */

/** The sheet. Generous margins so the full-bleed bars have something to bleed
 *  past — a poster needs an edge to run off. */
const SHELL = "mx-auto w-full max-w-[1400px] px-5 sm:px-8 lg:px-14 xl:px-20";

/** Display face. Tightly tracked at every size; this is the whole voice. */
const GROTESK = "font-[family-name:var(--font-grotesk)]";

/** Apparatus face. Labels, numbers, locators, scores, stack — anything that
 *  is machinery rather than speech. */
const MONO = "font-[family-name:var(--font-jetbrains)]";

/** The running label, in the apparatus face at printer's-slug tracking. */
const LABEL = `${MONO} text-[10px] font-medium uppercase leading-none tracking-[0.22em] sm:text-[11px]`;

/** Body measure. A poster still has to be read. */
const MEASURE = "max-w-[62ch]";

const BODY = "text-[15px] leading-[1.6] sm:text-[16px] sm:leading-[1.62]";

/* ══════════════════════════════════════════════════════════════════════════
   MOTION
   ══════════════════════════════════════════════════════════════════════════

   Press moves like a press. Blocks accelerate into position and stop dead
   against the platen; the second plate lands long, recoils, and settles a few
   pixels out of register; rules wipe across at speed. Nothing floats, nothing
   drifts, nothing eases gently to a halt.

   Only transform, opacity and a scalar counter are touched, so every frame
   stays on the compositor. */

/** Accelerating into a hard stop — the impact. Nothing decelerates here. */
const EASE_STRIKE = "inQuart";
/** A shorter strike for supporting matter, so it never outruns the block. */
const EASE_KNOCK = "inCubic";
/** Snap with a little overrun, for stamps and struck rules. */
const EASE_SNAP = "outBack(2.4)";
/** Mechanical, symmetrical, fast in the middle — a carriage returning. */
const EASE_CARRIAGE = "inOutQuint";
/** The plate recoiling into register. Stiff and barely damped, on purpose. */
const SPRING_REGISTER = spring({ stiffness: 240, damping: 13, mass: 1 });

/** Impact durations. Short — a slam that takes a second is a slide. */
const T_STRIKE = 300;
const T_KNOCK = 240;
const T_WIPE = 420;
const T_COUNT = 900;

/**
 * Fires once, when the element's top has risen `offset` px above the fold.
 *
 * anime splits an `enter` string as `[container, target]`, so this reads as
 * "the container's bottom, less the offset, meets the target's top".
 * `sync: 'play'` registers an enter method and no leave method, which is what
 * makes a reveal one-way — the default `'play pause'` would stall an
 * animation the moment its element scrolled back out of view.
 */
function enters(target: HTMLElement | SVGElement, offset = 90): ScrollObserver {
  return onScroll({ target, enter: `bottom-=${offset} top`, sync: "play", repeat: false });
}

type FigureFormat = {
  value: number;
  decimals: number;
  prefix: string;
  suffix: string;
};

/**
 * Reads a cell and reports whether there is anything to count to.
 *
 * `content.results` ships em dashes on purpose and counting an em dash up
 * from zero is nonsense, so `null` lets the caller pick the honest treatment
 * instead. Deliberately tolerant of the shapes real results arrive in —
 * `0.87`, `94%`, `412 ms` — so none of this needs revisiting when they land.
 */
function parseFigure(raw: string): FigureFormat | null {
  const match = /^([^\d+.-]*)([+-]?\d+(?:\.\d+)?)(.*)$/.exec(raw.trim());
  if (!match) return null;

  const [, prefix, digits, suffix] = match;
  const value = Number(digits);
  if (!Number.isFinite(value)) return null;

  const point = digits.indexOf(".");
  return {
    value,
    decimals: point === -1 ? 0 : digits.length - point - 1,
    prefix,
    suffix,
  };
}

function formatFigure(value: number, format: FigureFormat): string {
  return `${format.prefix}${value.toFixed(format.decimals)}${format.suffix}`;
}

function setupPressMotion(scope: Scope): void {
  // The stylesheet hides the entrance state behind the same media query, so
  // when motion is off nothing was ever hidden and there is nothing to
  // restore. Returning here leaves a fully composed, fully readable page.
  if (prefersReducedMotion()) return;

  // `Scope['root']` is `Document | DOMTarget`; every member is a ParentNode,
  // but the union defeats the generic overload of `querySelectorAll`.
  const root = scope.root as ParentNode;

  /**
   * Everything the CSS gate hid must be handed to a specific animation.
   * Whatever is not gets swept at the end of this function — a missed hook
   * should cost a reveal, never leave a paragraph invisible.
   */
  const claimed = new Set<Element>();

  function claim<T extends Element>(elements: T[]): T[] {
    for (const element of elements) claimed.add(element);
    return elements;
  }

  const all = (selector: string, within: ParentNode = root): HTMLElement[] =>
    Array.from(within.querySelectorAll<HTMLElement>(selector));

  const hook = (name: string, within: ParentNode = root): HTMLElement[] =>
    all(`[data-reveal="${name}"]`, within);

  /**
   * Where a plate comes to rest, and how far the page is tilted, read back
   * out of the cascade.
   *
   * Both live in `PRESS_CSS` and the drops change at the `sm` breakpoint;
   * resolving them here rather than repeating the numbers is what stops an
   * animation landing three pixels away from where the stylesheet says the
   * plate lives. One read at setup, never per frame.
   */
  const sheet = getComputedStyle(root instanceof Element ? root : document.documentElement);
  const drops: Record<string, number> = {
    sm: Number.parseFloat(sheet.getPropertyValue("--pr-drop-sm")) || 4,
    md: Number.parseFloat(sheet.getPropertyValue("--pr-drop")) || 8,
    lg: Number.parseFloat(sheet.getPropertyValue("--pr-drop-lg")) || 12,
  };
  const tilt = Number.parseFloat(sheet.getPropertyValue("--pr-tilt")) || -2.5;
  const restingDrop = (element: HTMLElement) => drops[element.dataset.drop ?? "md"] ?? drops.md;

  /**
   * A plate landing.
   *
   * It arrives from well past its mark and springs back to a resting position
   * that is deliberately *not* zero — the plate settles a few pixels out of
   * register and stays there, which is the whole conceit of the page.
   */
  function land(plate: HTMLElement, autoplay: ScrollObserver | boolean) {
    const rest = restingDrop(plate);
    animate(plate, {
      x: [rest + 26, rest],
      y: [rest + 26, rest],
      opacity: [0, 1],
      duration: 620,
      ease: SPRING_REGISTER,
      autoplay,
    });
  }

  /** A stamp being struck: down from oversize, past the angle, back onto it. */
  function strike(stamp: HTMLElement, autoplay: ScrollObserver | boolean, delay = 0) {
    animate(stamp, {
      opacity: [0, 1],
      scale: [1.7, 1],
      rotate: [`${tilt + 9}deg`, `${tilt}deg`],
      duration: 520,
      ease: EASE_SNAP,
      delay,
      autoplay,
    });
  }

  /* ── The front page ─────────────────────────────────────────────────── */

  const headline = root.querySelector<HTMLElement>('[data-reveal="headline"]');

  if (headline) {
    // Split into lines, wrap each in its own clipping box, push every line
    // below its mask, and only then let the headline become visible — all
    // three synchronously inside one frame, so the un-masked state is never
    // painted. `splitText` registers on the active scope, so `scope.revert()`
    // restores the original markup when the page unmounts.
    //
    // Driven from `addEffect` because the splitter measures on its own
    // schedule: read straight after construction, `split.lines` is still empty,
    // which silently skips the whole reveal. The effect also re-runs when a
    // resize reflows the lines.
    const split = splitText(headline, { lines: { wrap: "hidden" }, accessible: true });
    claimed.add(headline);

    split.addEffect(() => {
      // Re-queried rather than read off `split.lines`: those entries can be
      // detached nodes from an earlier pass, which animate to nothing, silently.
      const lines = Array.from(headline.querySelectorAll<HTMLElement>("[data-line]"));
      if (lines.length === 0) return;

      utils.set(lines, { y: "110%" });
      utils.set(headline, { opacity: 1 });

      // Each line is thrown up into its mask and stopped against the top of it.
      // 64ms apart: fast enough to read as one impact with a ripple through it
      // rather than as four separate arrivals.
      animate(lines, {
        y: ["110%", "0%"],
        duration: 380,
        delay: stagger(64, { start: 260 }),
        ease: EASE_STRIKE,
      });
    });
  }

  const overture = createTimeline({ defaults: { ease: EASE_STRIKE, duration: T_STRIKE } });

  overture.add(claim(hook("masthead")), { opacity: [0, 1], y: [-14, 0], duration: T_KNOCK }, 0);
  overture.add(
    claim(hook("slugline")),
    { opacity: [0, 1], scaleX: [0, 1], duration: T_WIPE, ease: EASE_CARRIAGE },
    60,
  );
  overture.add(
    claim(hook("eyebrow")),
    // `tilt` resolved from the cascade, not spelled as var(--pr-tilt): anime.js
    // interpolates parsed values and would land the eyebrow flat.
    { opacity: [0, 1], scale: [1.5, 1], rotate: ["6deg", `${tilt}deg`], ease: EASE_SNAP, duration: 520 },
    180,
  );

  const headlineGhost = hook("headline-ghost")[0];
  if (headlineGhost) {
    // The second plate, arriving after the black and overrunning it before it
    // settles into permanent misregistration.
    const rest = restingDrop(headlineGhost);
    animate(claim([headlineGhost]), {
      x: [rest + 34, rest],
      y: [rest + 34, rest],
      opacity: [0, 1],
      duration: 760,
      ease: SPRING_REGISTER,
      delay: 300,
    });
  }

  overture
    .add(claim(hook("standfirst")), { opacity: [0, 1], y: [22, 0], ease: EASE_KNOCK, duration: T_KNOCK }, 620)
    .add(
      claim(hook("cta")),
      { opacity: [0, 1], y: [26, 0], ease: EASE_KNOCK, duration: T_KNOCK, delay: stagger(70) },
      700,
    )
    .add(
      claim(hook("colophon")),
      { opacity: [0, 1], x: [22, 0], ease: EASE_KNOCK, duration: T_KNOCK, delay: stagger(55) },
      780,
    );

  for (const plate of claim(all('[data-hero] [data-reveal="plate"]'))) {
    land(plate, true);
  }

  /* ── Section furniture ──────────────────────────────────────────────── */

  for (const section of all("[data-section]")) {
    const observer = () => enters(section);

    const play = (targets: HTMLElement[], params: AnimationParams) => {
      if (targets.length === 0) return;
      animate(claim(targets), { ...params, autoplay: observer() });
    };

    // The slug bar wipes across first, the way a rule is pulled, then the
    // title strikes down into it and the deck knocks in behind.
    play(hook("bar", section), {
      opacity: [0, 1],
      scaleX: [0, 1],
      duration: T_WIPE,
      ease: EASE_CARRIAGE,
    });
    play(hook("title", section), {
      opacity: [0, 1],
      y: [-30, 0],
      duration: T_STRIKE,
      ease: EASE_STRIKE,
      delay: 150,
    });
    play(hook("deck", section), {
      opacity: [0, 1],
      y: [24, 0],
      duration: T_KNOCK,
      ease: EASE_KNOCK,
      delay: stagger(70, { start: 260 }),
    });
  }

  /* ── Blocks ─────────────────────────────────────────────────────────── */

  // A block is thrown up from below and stopped flat. `data-group` staggers a
  // run of them so a grid prints row by row rather than all at once.
  for (const group of all("[data-group]")) {
    const blocks = claim(all(":scope > *", group));
    if (blocks.length === 0) continue;

    // Assigned rather than spread: spreading a conditional `{ x } | {}`
    // widens `x` to include `undefined`, which AnimationParams rejects.
    const params: AnimationParams = {
      opacity: [0, 1],
      y: [34, 0],
      duration: T_STRIKE,
      ease: EASE_STRIKE,
      delay: stagger(70),
      autoplay: enters(group, 70),
    };
    if (group.dataset.group === "skew") params.x = [-18, 0];

    animate(blocks, params);
  }

  for (const block of claim(hook("block"))) {
    animate(block, {
      opacity: [0, 1],
      y: [34, 0],
      duration: T_STRIKE,
      ease: EASE_STRIKE,
      autoplay: enters(block, 70),
    });
  }

  // Plates outside the hero land when their own block does.
  for (const plate of claim(all('[data-reveal="plate"]:not([data-hero] *)'))) {
    land(plate, enters(plate, 70));
  }

  /* ── Stamps ─────────────────────────────────────────────────────────── */

  // Struck rather than faded: down from oversize, past the mark, back onto it.
  // `strike` resolves the resting angle off the cascade at setup: anime.js
  // cannot interpolate a raw `var(--pr-tilt)`, which would land the stamp flat.
  for (const stamp of claim(hook("stamp"))) {
    strike(stamp, enters(stamp, 60));
  }

  /* ── Rules ──────────────────────────────────────────────────────────── */

  for (const rule of claim(hook("wipe"))) {
    animate(rule, {
      opacity: [0, 1],
      scaleX: [0, 1],
      duration: T_WIPE,
      ease: EASE_CARRIAGE,
      autoplay: enters(rule, 60),
    });
  }

  const bars = claim(hook("strata"));
  if (bars.length > 0) {
    animate(bars, {
      opacity: [0, 1],
      scaleX: [0, 1],
      duration: 380,
      ease: EASE_STRIKE,
      delay: stagger(110),
      autoplay: enters(bars[0], 60),
    });
  }

  const scores = claim(hook("score"));
  if (scores.length > 0) {
    animate(scores, {
      opacity: [0, 1],
      scaleX: [0, 1],
      duration: 340,
      ease: EASE_STRIKE,
      delay: stagger(90),
      autoplay: enters(scores[0], 40),
    });
  }

  /* ── Figures ────────────────────────────────────────────────────────── */

  for (const cell of claim(hook("figure"))) {
    const raw = cell.dataset.value ?? cell.textContent ?? "";
    const format = parseFigure(raw);

    if (!format) {
      // Nothing to count to. The placeholder still arrives with its row
      // instead of sitting there pre-composed, but it does not pretend to be
      // a measurement in progress.
      animate(cell, {
        opacity: [0, 1],
        y: [18, 0],
        duration: T_KNOCK,
        ease: EASE_KNOCK,
        autoplay: enters(cell, 60),
      });
      continue;
    }

    // Real figures roll up on their own drum. Every cell is a fixed-width,
    // tabular-figure column, so digits changing width mid-count cannot move
    // a rule.
    const counter = { current: 0 };
    cell.textContent = formatFigure(0, format);

    animate(cell, {
      opacity: [0, 1],
      y: [18, 0],
      duration: T_KNOCK,
      ease: EASE_KNOCK,
      autoplay: enters(cell, 60),
    });
    animate(counter, {
      current: format.value,
      duration: T_COUNT,
      ease: EASE_CARRIAGE,
      autoplay: enters(cell, 60),
      onUpdate: () => {
        cell.textContent = formatFigure(counter.current, format);
      },
    });
  }

  /* ── Backstop ───────────────────────────────────────────────────────── */

  // Anything the gate hid that no animation above claimed. Reaching this loop
  // is a bug in the markup, but an unread paragraph is a far worse outcome
  // than a missing reveal, so it resolves in favour of the reader.
  for (const orphan of Array.from(root.querySelectorAll("[data-reveal], [data-group] > *"))) {
    if (claimed.has(orphan)) continue;
    if (orphan instanceof HTMLElement || orphan instanceof SVGElement) {
      orphan.style.opacity = "1";
    }
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   ORNAMENT
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * A halftone screen — real tiled geometry rather than a gradient, because a
 * dot screen is dots. Decorative and inert; density comes from `.pr-screen`.
 *
 * Ids are literals rather than `useId` because exactly one Press mounts at a
 * time and `url(#…)` references are easier to read when they are readable.
 */
function DotScreen({ id, tone = "ink" }: { id: string; tone?: "ink" | "stock" }) {
  const fill = tone === "ink" ? "var(--pr-ink)" : "var(--pr-stock)";
  return (
    <svg
      aria-hidden
      className="pr-screen pointer-events-none absolute inset-0 h-full w-full"
      preserveAspectRatio="none"
    >
      <defs>
        <pattern id={id} width="7" height="7" patternUnits="userSpaceOnUse">
          <circle cx="1.75" cy="1.75" r="1.35" fill={fill} />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${id})`} />
    </svg>
  );
}

/** A registration crosshair. The mark a printer lines two plates up on, and
 *  the reason the word "misregistration" means anything here. */
// Defaults to currentColor so a mark sitting on a flag block follows whatever
// that block set its type to, instead of staying ink and vanishing when a
// palette inverts.
function RegMark({
  tone = "current",
  className,
}: {
  tone?: "current" | "ink" | "flag";
  className?: string;
}) {
  const stroke =
    tone === "ink" ? "var(--pr-ink)" : tone === "flag" ? "var(--pr-flag)" : "currentColor";
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className={twMerge("block h-4 w-4 shrink-0", className)}
      shapeRendering="geometricPrecision"
    >
      <circle cx="12" cy="12" r="6" fill="none" stroke={stroke} strokeWidth="2" />
      <line x1="12" y1="0" x2="12" y2="5" stroke={stroke} strokeWidth="2" />
      <line x1="12" y1="19" x2="12" y2="24" stroke={stroke} strokeWidth="2" />
      <line x1="0" y1="12" x2="5" y2="12" stroke={stroke} strokeWidth="2" />
      <line x1="19" y1="12" x2="24" y2="12" stroke={stroke} strokeWidth="2" />
    </svg>
  );
}

/** A struck-through mark for the things this deliberately does not do. Two
 *  hard strokes, no curves — a line drawn through an entry on a list. */
function StrikeMark() {
  return (
    <svg aria-hidden viewBox="0 0 24 24" className="block h-full w-full" shapeRendering="crispEdges">
      <line x1="3" y1="3" x2="21" y2="21" stroke="var(--pr-on-flag)" strokeWidth="3" />
      <line x1="21" y1="3" x2="3" y2="21" stroke="var(--pr-on-flag)" strokeWidth="3" />
    </svg>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   BLOCKS
   ══════════════════════════════════════════════════════════════════════════ */

const DROP_CLASS = {
  sm: "pr-ghost-sm",
  md: "pr-ghost",
  lg: "pr-ghost-lg",
} as const;

type Drop = keyof typeof DROP_CLASS;

/**
 * A two-pass print: a flat plate behind, the sheet on top, a few pixels out
 * of register.
 *
 * This replaces `box-shadow` everywhere on the page. A shadow implies a light
 * source and a raised object; a misregistered plate implies a second run
 * through the press, which is the thing being said.
 */
function Plate({
  tone = "flag",
  drop = "md",
  className,
  sheetClassName,
  children,
}: {
  tone?: "flag" | "ink" | "proof";
  drop?: Drop;
  className?: string;
  sheetClassName?: string;
  children: ReactNode;
}) {
  return (
    <div className={twMerge("relative", className)}>
      <span
        aria-hidden
        data-reveal="plate"
        data-drop={drop}
        className={clsx(
          "pr-edge pointer-events-none absolute inset-0",
          DROP_CLASS[drop],
          tone === "flag" && "bg-[var(--pr-flag)]",
          tone === "ink" && "bg-[var(--pr-ink)]",
          tone === "proof" && "bg-[var(--pr-proof)]",
        )}
      />
      <div className={twMerge("pr-edge relative bg-[var(--pr-sheet)]", sheetClassName)}>
        {children}
      </div>
    </div>
  );
}

/**
 * The section slug: a full-bleed black bar with the number, the name and a
 * registration mark. It runs off both edges of the sheet because a poster's
 * furniture is trimmed, not centred.
 */
function SlugBar({ number, name, note }: { number: string; name: string; note: string }) {
  return (
    <div
      data-reveal="bar"
      className="relative origin-left overflow-hidden bg-[var(--pr-ink)] text-[var(--pr-stock)]"
    >
      <div className={clsx(SHELL, "flex items-center gap-3 py-2.5 sm:gap-5 sm:py-3")}>
        <span className={clsx(MONO, "text-[11px] font-bold tabular-nums sm:text-[13px]")}>
          {number}
        </span>
        <span aria-hidden className="h-4 w-px shrink-0 bg-[var(--pr-stock)] opacity-40" />
        <span className={clsx(LABEL, "truncate")}>{name}</span>
        <span
          className={clsx(
            MONO,
            "ml-auto hidden text-[10px] tracking-[0.18em] whitespace-nowrap uppercase opacity-70 md:inline",
          )}
        >
          {note}
        </span>
        <RegMark tone="flag" className="ml-auto h-3.5 w-3.5 md:ml-4" />
      </div>
    </div>
  );
}

/** A rotated, bordered label. One tilt value, from the token block. */
function Stamp({
  tone = "flag",
  reveal = true,
  className,
  children,
}: {
  tone?: "flag" | "ink" | "proof" | "open";
  reveal?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      data-reveal={reveal ? "stamp" : undefined}
      className={twMerge(
        clsx(
          LABEL,
          "pr-edge pr-stamp inline-flex origin-center items-center gap-2 px-2.5 py-1.5 sm:px-3 sm:py-2",
          tone === "flag" && "bg-[var(--pr-flag)] text-[var(--pr-on-flag)]",
          tone === "ink" && "bg-[var(--pr-ink)] text-[var(--pr-stock)]",
          tone === "proof" && "bg-[var(--pr-proof)] text-[var(--pr-stock)]",
          tone === "open" && "bg-[var(--pr-sheet)] text-[var(--pr-ink)]",
        ),
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A hard-edged token: nav item, suggestion, stack entry. Small, bordered,
 *  square, in the apparatus face. */
function Chip({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <span
      className={twMerge(
        clsx(MONO, "pr-edge inline-flex items-center gap-2 bg-[var(--pr-sheet)] px-2.5 py-1.5 text-[11px] tracking-[0.08em] uppercase"),
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * A section.
 *
 * The slug bar bleeds full width; everything under it sits on the same twelve
 * columns. The asymmetry is fixed rather than alternating — title heavy on
 * the left across seven columns, apparatus in a plated block on the right
 * across four, column eight left empty as the gutter that makes the split
 * read as a decision.
 */
function Section({
  id,
  number,
  slug,
  note,
  title,
  deck,
  aside,
  asideLabel,
  children,
}: {
  id: string;
  number: string;
  slug: string;
  note: string;
  title: string;
  deck: string;
  aside: ReactNode;
  asideLabel: string;
  children: ReactNode;
}) {
  return (
    <section
      id={id}
      data-section={id}
      aria-labelledby={`${id}-title`}
      className="scroll-mt-4 pt-14 pb-16 sm:pt-20 sm:pb-24 lg:pt-24 lg:pb-32"
    >
      <SlugBar number={number} name={slug} note={note} />

      <div className={clsx(SHELL, "mt-10 sm:mt-14")}>
        <div className="grid grid-cols-1 gap-x-8 gap-y-10 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <h2
              id={`${id}-title`}
              data-reveal="title"
              className={clsx(
                GROTESK,
                "text-[length:var(--pr-display)] leading-[0.86] font-bold tracking-[-0.04em] uppercase",
              )}
            >
              {title}
            </h2>
            <p
              data-reveal="deck"
              className={clsx(MEASURE, BODY, "mt-6 text-[var(--pr-ink-2)] sm:mt-8")}
            >
              {deck}
            </p>
          </div>

          <div className="lg:col-span-4 lg:col-start-9">
            <Plate tone="ink" drop="sm">
              <div className="p-5 sm:p-6">
                <p className={clsx(LABEL, "text-[var(--pr-flag-ink)]")}>{asideLabel}</p>
                <div className="pr-edge-t mt-4 pt-4 text-[13px] leading-[1.6] text-[var(--pr-ink-2)]">
                  {aside}
                </div>
              </div>
            </Plate>
          </div>
        </div>

        <div className="mt-14 sm:mt-18 lg:mt-24">{children}</div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   THE TRANSCRIPT
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * A citation, set as a printer's docket line rather than a confidence badge:
 * source, locator, score, and a hard bar the score's width. The bar is a flat
 * fill in the flag colour, which is legal — it is a mark, not type.
 */
function CitationLine({ index, citation }: { index: number; citation: Citation }) {
  return (
    <li className="pr-edge-t grid grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 py-3 sm:grid-cols-[2.25rem_minmax(0,1fr)_6rem_auto] sm:gap-x-4">
      <span
        className={clsx(
          MONO,
          "pr-edge flex h-6 w-6 items-center justify-center bg-[var(--pr-flag)] text-[11px] font-bold tabular-nums text-[var(--pr-on-flag)] sm:h-7 sm:w-7 sm:text-[12px]",
        )}
      >
        {index}
      </span>
      <span className={clsx(MONO, "truncate text-[12px] font-medium sm:text-[13px]")}>
        {citation.source}
      </span>
      <span
        className={clsx(
          MONO,
          "col-start-2 text-[11px] text-[var(--pr-ink-3)] sm:col-start-3 sm:row-start-1 sm:text-[12px]",
        )}
      >
        {citation.locator}
      </span>
      <span className="col-start-3 row-start-1 flex items-center gap-2 sm:col-start-4">
        <span aria-hidden className="pr-edge hidden h-3 w-20 bg-[var(--pr-sheet)] sm:block">
          <span
            data-reveal="score"
            className="block h-full origin-left bg-[var(--pr-flag)]"
            style={{ width: `${Math.round(citation.score * 100)}%` }}
          />
        </span>
        <span className={clsx(MONO, "text-[12px] font-bold tabular-nums sm:text-[13px]")}>
          {citation.score.toFixed(2)}
        </span>
      </span>
    </li>
  );
}

/** A turn label — who is speaking, in the apparatus face, above the block. */
function TurnLabel({ children, tone = "ink" }: { children: ReactNode; tone?: "ink" | "flag" }) {
  return (
    <p
      className={clsx(
        LABEL,
        "mb-3 flex items-center gap-2.5",
        tone === "flag" ? "text-[var(--pr-flag-ink)]" : "text-[var(--pr-ink-3)]",
      )}
    >
      <span
        aria-hidden
        className={clsx(
          "block h-2.5 w-2.5",
          tone === "flag" ? "bg-[var(--pr-flag)]" : "bg-[var(--pr-ink)]",
        )}
      />
      {children}
    </p>
  );
}

/**
 * The refusal.
 *
 * The loudest object on the page, and deliberately so: a solid flag-coloured
 * block, over-plated, with the finding set at display size and a struck stamp
 * across the corner. It is not a greyed-out error — it is the one behaviour
 * this project is built to have, printed the way a headline is printed.
 *
 * It keeps the reference apparatus of a real answer, which is what turns an
 * empty citation list into the point being made rather than something
 * missing. Everything on the flag is set in --pr-ink at 6.0:1; nothing here
 * is stock-on-orange, which would be 2.6:1 and illegible.
 */
function Refusal({ text }: { text: string }) {
  // Guarded so an edit that drops the full stop degrades to one block of copy
  // rather than losing the sentence.
  const stop = text.indexOf(". ");
  const finding = stop === -1 ? text : text.slice(0, stop + 1);
  const reasoning = stop === -1 ? "" : text.slice(stop + 2);

  return (
    <div className="lg:col-span-11 lg:col-start-2">
      <TurnLabel tone="flag">Refusal — by design</TurnLabel>

      <div className="relative">
        <span
          aria-hidden
          data-reveal="plate"
          data-drop="lg"
          className="pr-edge-2 pr-ghost-lg pointer-events-none absolute inset-0 bg-[var(--pr-ink)]"
        />

        <div
          data-reveal="block"
          className="pr-edge-2 relative overflow-hidden bg-[var(--pr-flag)] text-[var(--pr-on-flag)]"
        >
          <DotScreen id="pr-screen-refusal" />

          <div className="relative p-6 sm:p-9 lg:p-12">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <span className={clsx(LABEL, "pr-edge bg-[var(--pr-ink)] px-3 py-2 text-[var(--pr-stock)]")}>
                No answer returned
              </span>
              <Stamp tone="ink" className="shrink-0">
                Not in the corpus
              </Stamp>
            </div>

            <p
              className={clsx(
                GROTESK,
                "mt-8 max-w-[16ch] text-[length:var(--pr-lead)] leading-[0.94] font-bold tracking-[-0.035em] uppercase sm:mt-10",
              )}
            >
              {finding}
            </p>

            {reasoning ? (
              <p className={clsx(MEASURE, "mt-6 text-[15px] leading-[1.62] sm:text-[16.5px]")}>
                {reasoning}
              </p>
            ) : null}

            <div className="pr-edge-2 mt-9 border-r-0 border-b-0 border-l-0 pt-5 sm:mt-12">
              <div className="grid grid-cols-1 gap-x-8 gap-y-5 sm:grid-cols-12">
                <p className={clsx(LABEL, "sm:col-span-3")}>Sources cited</p>
                <p className="text-[13.5px] leading-[1.6] sm:col-span-9 sm:text-[14px]">
                  <span className={clsx(MONO, "font-bold")}>NONE.</span> Retrieval returned nothing
                  above threshold, so the model was never called. There is no passage to cite
                  because there was no passage — and an answer assembled without one is the failure
                  this whole system is arranged to avoid.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <p className={clsx(MEASURE, "mt-6 text-[13.5px] leading-[1.6] text-[var(--pr-ink-2)] sm:mt-8")}>
        Any RAG demo can show a confident, cited answer. Declining one the corpus cannot support is
        the behaviour almost nobody ships, which is why it is printed at this size rather than
        buried in the docs.
      </p>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   RESULTS
   ══════════════════════════════════════════════════════════════════════════ */

/**
 * A metric value in a reserved cell.
 *
 * The column is fixed-width and tabular so `0.87`, `94%` and `412 ms` all
 * drop into the space the em dash is already holding open — no reflow when
 * the evaluation finally runs. Unmeasured cells are set in the blueline: on a
 * press a blue proof is what you pull *before* the real run, which is exactly
 * the status of every number here.
 */
function MetricValue({ value }: { value: string }) {
  const measured = parseFigure(value) !== null;

  return (
    <span className="flex w-[7.5rem] shrink-0 items-baseline justify-end gap-1.5 sm:w-[9.5rem]">
      <span
        data-reveal="figure"
        data-value={value}
        className={clsx(
          MONO,
          "text-[length:var(--pr-figure)] leading-none font-bold tracking-[-0.04em] tabular-nums",
          measured ? "text-[var(--pr-ink)]" : "text-[var(--pr-proof)]",
        )}
      >
        {value}
      </span>
      {measured ? null : (
        <>
          <span aria-hidden className={clsx(MONO, "text-[11px] font-bold text-[var(--pr-proof)]")}>
            PR
          </span>
          <span className="sr-only">proof only, not yet measured</span>
        </>
      )}
    </span>
  );
}

/** One of the two measurement blocks. Rows are ruled, not striped; the value
 *  column is the same width in both so the pair reads as one table. */
function MetricBlock({
  index,
  title,
  question,
  rows,
}: {
  index: string;
  title: string;
  question: string;
  rows: readonly Metric[];
}) {
  return (
    <Plate tone="flag" drop="md" className="h-full">
      <figure className="m-0 h-full">
        <figcaption className="pr-edge-b-2 bg-[var(--pr-ink)] px-5 py-4 text-[var(--pr-stock)] sm:px-7 sm:py-5">
          <div className="flex items-center gap-3">
            <span className={clsx(MONO, "text-[11px] font-bold tabular-nums")}>{index}</span>
            <span aria-hidden className="h-3.5 w-px bg-[var(--pr-stock)] opacity-40" />
            <h3
              className={clsx(
                GROTESK,
                "text-[1.375rem] leading-none font-bold tracking-[-0.03em] uppercase sm:text-[1.75rem]",
              )}
            >
              {title}
            </h3>
          </div>
          <p className="mt-2.5 text-[12.5px] leading-[1.5] opacity-80 sm:text-[13px]">{question}</p>
        </figcaption>

        <ul data-group="rows" className="px-5 sm:px-7">
          {rows.map((row) => (
            <li
              key={row.label}
              className="flex items-center justify-between gap-4 border-b-[length:var(--pr-rule)] border-dashed border-[var(--pr-ink-3)] py-5 last:border-b-0 sm:py-6"
            >
              <span className="min-w-0">
                <span className={clsx(MONO, "block text-[13px] font-bold sm:text-[14px]")}>
                  {row.label}
                </span>
                <span className="mt-1.5 block text-[12px] leading-[1.45] text-[var(--pr-ink-2)] sm:text-[12.5px]">
                  {row.caption}
                </span>
              </span>
              <MetricValue value={row.value} />
            </li>
          ))}
        </ul>
      </figure>
    </Plate>
  );
}

/**
 * The provisional notice.
 *
 * Driven by `METRICS_ARE_PLACEHOLDER`, so wiring the real evaluation output
 * into `content.ts` removes this block on its own — nobody has to remember to
 * take the disclaimer down, which is exactly how fake numbers survive review.
 */
function ProofNotice() {
  return (
    <div className="pr-edge-proof relative overflow-hidden bg-[var(--pr-sheet)]">
      <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:gap-6 sm:p-6">
        <span
          className={clsx(
            LABEL,
            "pr-edge-proof inline-flex shrink-0 items-center gap-2 self-start bg-[var(--pr-proof)] px-3 py-2 text-[var(--pr-stock)]",
          )}
        >
          Blueline proof
        </span>
        <p className="text-[13px] leading-[1.6] text-[var(--pr-ink-2)] sm:text-[13.5px]">
          The evaluation run has not been executed. Every cell marked{" "}
          <span className={clsx(MONO, "font-bold text-[var(--pr-proof)]")}>PR</span> is empty on
          purpose rather than plausible — a placeholder number is the kind that survives review and
          ends up in a README. Each column is already sized for the figure that will replace it, so
          nothing on this page moves when the numbers land.
        </p>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   PAGE FURNITURE
   ══════════════════════════════════════════════════════════════════════════ */

/** Printer's slug: the tagline repeated across a rule and trimmed at both
 *  edges. Static — a marquee that loops forever is a battery cost and a
 *  distraction, and the texture is the point, not the movement. */
function SlugLine() {
  return (
    <div
      data-reveal="slugline"
      aria-hidden
      className="pr-edge-b origin-left overflow-hidden bg-[var(--pr-flag)] text-[var(--pr-on-flag)]"
    >
      <div className="flex items-center gap-6 py-2 whitespace-nowrap">
        {Array.from({ length: 10 }, (_, index) => (
          <Fragment key={index}>
            <span
              className={clsx(
                MONO,
                "text-[10px] font-bold tracking-[0.24em] text-[var(--pr-on-flag)] uppercase",
              )}
            >
              {content.brand.tagline}
            </span>
            <RegMark className="h-3 w-3" />
          </Fragment>
        ))}
      </div>
    </div>
  );
}

/** The pipeline stages, derived rather than restated, so the running head
 *  cannot drift from the content model. */
const STAGE_COUNT = content.pipeline.length;
const QUESTION_COUNT = content.demo.messages.filter((message) => message.role === "user").length;
const MEASURE_COUNT = content.results.retrieval.length + content.results.generation.length;

/** 70 in ink, 20 in the flag because the unanswerable questions are the whole
 *  argument, 10 in the quiet grey. Bars are marks, not type, so the flag is
 *  legal here. */
const STRATA_FILL = ["bg-[var(--pr-ink)]", "bg-[var(--pr-flag)]", "bg-[var(--pr-ink-3)]"];
const STRATA_TEXT = ["text-[var(--pr-ink)]", "text-[var(--pr-flag-ink)]", "text-[var(--pr-ink-3)]"];

/** Which pipeline blocks drop below their row. A fixed two-step pattern —
 *  the sheets stack unevenly, but always by the same amount. */
const STAGE_OFFSET = ["", "lg:mt-14", "", "", "lg:mt-14", ""];

export default function LandingPage() {
  const root = useAnimeScope<HTMLDivElement>(setupPressMotion);

  return (
    <div ref={root} className={clsx("press min-h-screen", GROTESK)}>
      <style href="press-tokens" precedence="high">
        {PRESS_CSS}
      </style>

      {/* ── Masthead ──────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 bg-[var(--pr-stock)]">
        <SlugLine />

        <div data-reveal="masthead" className="pr-edge-b-2 bg-[var(--pr-stock)]">
          <div
            className={clsx(
              SHELL,
              "flex flex-wrap items-center gap-x-4 gap-y-3 py-3 sm:gap-x-6 sm:py-4",
            )}
          >
            <a
              href="#overview"
              className={clsx(
                "text-[1.5rem] leading-none font-bold tracking-[-0.05em] uppercase sm:text-[1.875rem]",
              )}
            >
              {content.brand.name}
            </a>

            <span aria-hidden className="hidden h-6 w-[3px] bg-[var(--pr-ink)] sm:block" />

            <p
              className={clsx(
                MONO,
                "hidden text-[11px] tracking-[0.1em] text-[var(--pr-ink-2)] uppercase md:block",
              )}
            >
              {content.brand.tagline}
            </p>

            <nav aria-label="Sections" className="ml-auto">
              <ul className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                {content.nav.map((item) => (
                  <li key={item.href}>
                    <a
                      href={item.href}
                      className={clsx(
                        MONO,
                        "pr-edge block bg-[var(--pr-sheet)] px-2.5 py-1.5 text-[10px] tracking-[0.12em] uppercase transition-colors duration-150 hover:bg-[var(--pr-flag)] hover:text-[var(--pr-on-flag)] sm:px-3 sm:py-2 sm:text-[11px]",
                      )}
                    >
                      {item.label}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </div>
      </header>

      <main>
        {/* ── The front page ──────────────────────────────────────────── */}
        <section
          id="overview"
          data-hero
          className="scroll-mt-4 pt-12 pb-16 sm:pt-16 sm:pb-24 lg:pt-20 lg:pb-32"
        >
          <div className={SHELL}>
            <div className="flex flex-wrap items-center gap-4 sm:gap-6">
              <span data-reveal="eyebrow" className="inline-block origin-center">
                <Stamp tone="flag" reveal={false}>
                  <RegMark className="h-3 w-3" />
                  {content.hero.eyebrow}
                </Stamp>
              </span>
              <span
                data-reveal="colophon"
                className={clsx(
                  MONO,
                  "text-[10px] tracking-[0.2em] text-[var(--pr-ink-3)] uppercase sm:text-[11px]",
                )}
              >
                {STAGE_COUNT} stages · {MEASURE_COUNT} measures · 100 labelled questions
              </span>
            </div>

            {/* The signature: a black plate and a flag plate that never quite
                line up. The orange copy is aria-hidden and carries no
                information — it is the one place the flag colour touches a
                glyph, which is legal precisely because nothing is being read
                from it. */}
            <div className="relative mt-9 sm:mt-12 lg:mt-14">
              <span
                aria-hidden
                data-reveal="headline-ghost"
                data-drop="lg"
                className={clsx(
                  "pr-ghost-lg pointer-events-none absolute inset-0 text-[length:var(--pr-mega)] leading-[0.82] font-bold tracking-[-0.045em] text-[var(--pr-flag)] uppercase",
                )}
              >
                {content.hero.headline}
              </span>
              <h1
                data-reveal="headline"
                className="pr-headline relative text-[length:var(--pr-mega)] leading-[0.82] font-bold tracking-[-0.045em] uppercase"
              >
                {content.hero.headline}
              </h1>
            </div>

            <div className="mt-10 grid grid-cols-1 gap-x-8 gap-y-10 sm:mt-14 lg:grid-cols-12">
              <div className="lg:col-span-7">
                <p
                  data-reveal="standfirst"
                  className={clsx(MEASURE, "text-[16px] leading-[1.6] sm:text-[18px] sm:leading-[1.58]")}
                >
                  {content.hero.subhead}
                </p>

                <div className="mt-9 flex flex-wrap items-center gap-4 sm:mt-11 sm:gap-6">
                  <a
                    data-reveal="cta"
                    href="#pipeline"
                    className={clsx(
                      "pr-edge-2 pr-key inline-flex items-center gap-3 bg-[var(--pr-flag)] px-5 py-3.5 text-[13px] font-bold tracking-[0.04em] text-[var(--pr-on-flag)] uppercase sm:px-7 sm:py-4 sm:text-[14px]",
                    )}
                  >
                    {content.hero.primaryCta}
                    <ArrowRight size={16} strokeWidth={3} aria-hidden />
                  </a>
                  <a
                    data-reveal="cta"
                    href="#results"
                    className={clsx(
                      "pr-edge-2 pr-key inline-flex items-center gap-3 bg-[var(--pr-sheet)] px-5 py-3.5 text-[13px] font-bold tracking-[0.04em] uppercase sm:px-7 sm:py-4 sm:text-[14px]",
                    )}
                  >
                    {content.hero.secondaryCta}
                    <ArrowDownRight size={16} strokeWidth={3} aria-hidden />
                  </a>
                </div>
              </div>

              {/* Contents, as a printer's docket. */}
              <nav aria-labelledby="docket-label" className="lg:col-span-4 lg:col-start-9">
                <Plate tone="ink" drop="md">
                  <div className="p-5 sm:p-6">
                    <p id="docket-label" className={clsx(LABEL, "text-[var(--pr-ink-3)]")}>
                      On this sheet
                    </p>
                    <ol className="mt-4">
                      {SHEET_CONTENTS.map((entry) => (
                        <li key={entry.id} data-reveal="colophon" className="pr-edge-t">
                          <a
                            href={`#${entry.id}`}
                            className="group flex items-center gap-3 py-2.5 transition-colors duration-150 hover:bg-[var(--pr-flag)] hover:text-[var(--pr-on-flag)]"
                          >
                            <span
                              className={clsx(
                                MONO,
                                "text-[11px] font-bold tabular-nums text-[var(--pr-flag-ink)] group-hover:text-[var(--pr-on-flag)]",
                              )}
                            >
                              {entry.number}
                            </span>
                            <span className="text-[13px] font-bold tracking-[-0.01em] uppercase">
                              {entry.slug}
                            </span>
                            <span
                              className={clsx(
                                MONO,
                                "ml-auto text-[10px] tracking-[0.1em] text-[var(--pr-ink-3)] uppercase group-hover:text-[var(--pr-on-flag)]",
                              )}
                            >
                              {entry.note}
                            </span>
                          </a>
                        </li>
                      ))}
                    </ol>
                  </div>
                </Plate>
              </nav>
            </div>
          </div>
        </section>

        {/* ── 01 The exchange ─────────────────────────────────────────── */}
        <Section
          id="transcript"
          number="01"
          slug="The exchange"
          note={`${QUESTION_COUNT} questions · 1 refusal`}
          title="Two questions, one answer"
          deck="A recorded exchange against an indexed corpus. The second question is the one that matters — adjacent to the material but absent from it, which is exactly the case a grounded system has to get right and almost every one gets wrong."
          asideLabel="How to read it"
          aside={
            <>
              Questions are set in the display face, answers on a printed sheet. Superscripts
              resolve to the passages the answer was actually built from, each with the retrieval
              score it scored. Nothing is paraphrased from outside them.
            </>
          }
        >
          {/* The prompt line. Not an input — nothing here is wired to a
              backend, and a live-looking text field that swallows keystrokes
              is a worse lie than a printed one. */}
          <div className="grid grid-cols-1 gap-x-8 gap-y-10 lg:grid-cols-12">
            <div className="lg:col-span-11 lg:col-start-2">
              <TurnLabel>Prompt line</TurnLabel>
              <Plate tone="flag" drop="sm">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-3 px-5 py-4 sm:px-6 sm:py-5">
                  <span className={clsx(MONO, "text-[11px] font-bold text-[var(--pr-flag-ink)]")}>
                    ASK&gt;
                  </span>
                  <span
                    className={clsx(
                      MONO,
                      "text-[13px] text-[var(--pr-ink-3)] sm:text-[14.5px]",
                    )}
                  >
                    {content.demo.placeholder}
                  </span>
                  <span aria-hidden className="block h-[1.1em] w-[0.55ch] bg-[var(--pr-ink)]" />
                  <span
                    className={clsx(
                      MONO,
                      "pr-edge ml-auto hidden bg-[var(--pr-flag)] px-3 py-1.5 text-[10px] font-bold tracking-[0.14em] text-[var(--pr-on-flag)] uppercase sm:block",
                    )}
                  >
                    Return
                  </span>
                </div>
              </Plate>

              <p className={clsx(LABEL, "mt-8 text-[var(--pr-ink-3)]")}>Also in this corpus</p>
              <ul data-group="skew" className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
                {content.demo.suggestions.map((suggestion, index) => (
                  <li key={suggestion} className="pr-edge flex gap-3 bg-[var(--pr-sheet)] p-4">
                    <span
                      className={clsx(
                        MONO,
                        "shrink-0 text-[11px] font-bold tabular-nums text-[var(--pr-flag-ink)]",
                      )}
                    >
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="text-[13px] leading-[1.45] text-[var(--pr-ink-2)]">
                      {suggestion}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            {content.demo.messages.map((message, index) => {
              if (message.role === "user") {
                return (
                  <div key={`turn-${index}`} className="lg:col-span-9">
                    <TurnLabel>Question {Math.floor(index / 2) + 1}</TurnLabel>
                    <div
                      data-reveal="block"
                      className="pr-edge-l bg-[var(--pr-stock)] pl-5 sm:pl-7"
                    >
                      <p
                        className={clsx(
                          "max-w-[22ch] text-[length:var(--pr-ask)] leading-[1.02] font-bold tracking-[-0.035em] uppercase",
                        )}
                      >
                        {message.text}
                      </p>
                    </div>
                  </div>
                );
              }

              if (message.role === "assistant") {
                return (
                  <div key={`turn-${index}`} className="lg:col-span-11 lg:col-start-2">
                    <TurnLabel>Answer — grounded</TurnLabel>
                    <Plate tone="ink" drop="md">
                      <div className="p-6 sm:p-8 lg:p-10">
                        <p className={clsx(MEASURE, BODY)}>
                          {message.text}
                          <sup className="ml-0.5 whitespace-nowrap">
                            {message.citations.map((citation, citationIndex) => (
                              <Fragment key={citation.locator}>
                                {citationIndex > 0 ? (
                                  <span className={clsx(MONO, "text-[10px] text-[var(--pr-ink-3)]")}>
                                    ,
                                  </span>
                                ) : null}
                                <a
                                  href={`#turn-${index}-ref-${citationIndex + 1}`}
                                  aria-label={`Reference ${citationIndex + 1}: ${citation.source}, ${citation.locator}`}
                                  className={clsx(
                                    MONO,
                                    "px-0.5 text-[11px] font-bold text-[var(--pr-flag-ink)] underline underline-offset-2",
                                  )}
                                >
                                  {citationIndex + 1}
                                </a>
                              </Fragment>
                            ))}
                          </sup>
                        </p>

                        <div className="pr-edge-t mt-8 pt-5 sm:mt-10">
                          <div className="flex items-baseline justify-between gap-4">
                            <p className={clsx(LABEL, "text-[var(--pr-ink-3)]")}>Sources cited</p>
                            <p
                              className={clsx(
                                MONO,
                                "text-[10px] tracking-[0.16em] text-[var(--pr-ink-3)] uppercase",
                              )}
                            >
                              Locator · Score
                            </p>
                          </div>
                          <ol className="mt-3">
                            {message.citations.map((citation, citationIndex) => (
                              <li
                                key={citation.locator}
                                id={`turn-${index}-ref-${citationIndex + 1}`}
                                className="scroll-mt-28"
                              >
                                <ol>
                                  <CitationLine index={citationIndex + 1} citation={citation} />
                                </ol>
                              </li>
                            ))}
                          </ol>
                        </div>
                      </div>
                    </Plate>
                  </div>
                );
              }

              return <Refusal key={`turn-${index}`} text={message.text} />;
            })}
          </div>
        </Section>

        {/* ── 02 The run ──────────────────────────────────────────────── */}
        <Section
          id="pipeline"
          number="02"
          slug="The run"
          note={`${STAGE_COUNT} stages, deterministic`}
          title={`${STAGE_COUNT} passes between a file and an answer`}
          deck="Every stage is deterministic and every stage has an explicit failure mode, so nothing is silently swallowed on the way through. A scanned PDF is rejected rather than indexed as an empty document; an empty retrieval is a valid outcome rather than an error."
          asideLabel="Why it repeats"
          aside={
            <>
              Chunk ids are stable and the index upserts on them, so re-ingesting a document updates
              it instead of duplicating it. An index that quietly grows on every upload cannot be
              measured twice.
            </>
          }
        >
          <ol data-group="stages" className="grid grid-cols-1 gap-6 sm:gap-8 lg:grid-cols-3">
            {content.pipeline.map((stage, index) => {
              // The last stage carries the flag: it is where the refusal
              // happens, and it is the only place in the pipeline where not
              // producing output is the correct result.
              const terminal = index === content.pipeline.length - 1;
              return (
                <li key={stage.id} className={clsx("h-full", STAGE_OFFSET[index])}>
                  <div className="relative h-full">
                    <span
                      aria-hidden
                      data-reveal="plate"
                      data-drop="md"
                      className={clsx(
                        "pr-edge pr-ghost pointer-events-none absolute inset-0",
                        terminal ? "bg-[var(--pr-ink)]" : "bg-[var(--pr-flag)]",
                      )}
                    />
                    <div
                      className={clsx(
                        "pr-edge relative flex h-full flex-col",
                        terminal
                          ? "bg-[var(--pr-flag)] text-[var(--pr-on-flag)]"
                          : "bg-[var(--pr-sheet)]",
                      )}
                    >
                      <div className="pr-edge-b flex items-center gap-3 px-5 py-3">
                        <span className={clsx(MONO, "text-[11px] font-bold tabular-nums")}>
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span aria-hidden className="h-3.5 w-px bg-current opacity-40" />
                        <span
                          className={clsx(
                            MONO,
                            "text-[10px] tracking-[0.18em] uppercase",
                            terminal ? "text-[var(--pr-on-flag)]" : "text-[var(--pr-ink-3)]",
                          )}
                        >
                          {stage.id}
                        </span>
                        {terminal ? <RegMark className="ml-auto h-3.5 w-3.5" /> : null}
                      </div>

                      <div className="flex flex-1 flex-col p-5 sm:p-6">
                        <h3
                          className={clsx(
                            "text-[length:var(--pr-stage)] leading-none font-bold tracking-[-0.035em] uppercase",
                          )}
                        >
                          {stage.title}
                        </h3>
                        <p
                          className={clsx(
                            "mt-4 text-[13px] leading-[1.58] sm:text-[13.5px]",
                            terminal ? "text-[var(--pr-on-flag)]" : "text-[var(--pr-ink-2)]",
                          )}
                        >
                          {stage.detail}
                        </p>
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
        </Section>

        {/* ── 03 Results ──────────────────────────────────────────────── */}
        <Section
          id="results"
          number="03"
          slug="Results"
          note={`${MEASURE_COUNT} measures · not yet run`}
          title={content.results.heading}
          deck={content.results.blurb}
          asideLabel="Status"
          aside={
            <>
              Not measured yet. These cells are deliberately empty rather than plausible, and the
              notice below is driven by the same flag that fills them — so wiring the real results
              in is what takes the disclaimer down.
            </>
          }
        >
          <div className="space-y-10 sm:space-y-12">
            {METRICS_ARE_PLACEHOLDER ? <ProofNotice /> : null}

            <div className="grid grid-cols-1 gap-8 sm:gap-10 lg:grid-cols-2 lg:gap-12">
              <MetricBlock
                index="TABLE 3.1"
                title="Retrieval"
                question="Does the right passage come back at all?"
                rows={content.results.retrieval}
              />
              <MetricBlock
                index="TABLE 3.2"
                title="Generation"
                question="Given the right passage, is the answer honest about it?"
                rows={content.results.generation}
              />
            </div>

            <p
              className={clsx(
                MEASURE,
                "text-[13.5px] leading-[1.62] text-[var(--pr-ink-2)] sm:text-[14px]",
              )}
            >
              Two tables rather than one composite score, because they answer different questions.
              Most RAG failures are retrieval failures blamed on the model, and a single number
              cannot tell you which half broke.
            </p>
          </div>
        </Section>

        {/* ── 04 The set ──────────────────────────────────────────────── */}
        <Section
          id="evaluation"
          number="04"
          slug="The set"
          note="100 questions, labelled first"
          title={content.evaluation.heading}
          deck="One hundred questions written against the corpus and labelled before any tuning happened. The proportions are the design: most of the set checks whether retrieval works, a fifth of it checks whether refusal does."
          asideLabel="What a label is"
          aside={
            <>
              Each answerable question carries the chunk that actually contains its answer. That is
              what lets retrieval be scored on its own, separately from whatever the model then does
              with the passage.
            </>
          }
        >
          {/* The strata, at true proportion. Marks, not type — the flag is
              legal here and carries the emphasis. */}
          <div aria-hidden className="pr-edge flex h-10 items-stretch overflow-hidden sm:h-14">
            {content.evaluation.points.map((point, index) => (
              <span
                key={point.n}
                data-reveal="strata"
                style={{ flexGrow: Number(point.n), flexBasis: 0 }}
                className={clsx(
                  "block origin-left",
                  STRATA_FILL[index],
                  index > 0 && "border-l-[length:var(--pr-rule)] border-[var(--pr-ink)]",
                )}
              />
            ))}
          </div>

          <ol data-group="strata-rows" className="mt-10 grid grid-cols-1 gap-6 sm:mt-12 sm:gap-8 lg:grid-cols-3">
            {content.evaluation.points.map((point, index) => (
              <li key={point.n} className="pr-edge-t pt-5 sm:pt-6">
                <span
                  data-reveal="figure"
                  data-value={point.n}
                  className={clsx(
                    "block text-[length:var(--pr-numeral)] leading-[0.8] font-bold tracking-[-0.06em] tabular-nums",
                    STRATA_TEXT[index],
                  )}
                >
                  {point.n}
                </span>
                <h3 className="mt-5 text-[1.125rem] leading-tight font-bold tracking-[-0.02em] uppercase sm:text-[1.25rem]">
                  {point.label}
                </h3>
                <p className="mt-3 text-[13px] leading-[1.58] text-[var(--pr-ink-2)] sm:text-[13.5px]">
                  {point.detail}
                </p>
              </li>
            ))}
          </ol>

          {/* The note, set as the finding it is. */}
          <div className="relative mt-14 sm:mt-20">
            <span
              aria-hidden
              data-reveal="plate"
              data-drop="lg"
              className="pr-edge-2 pr-ghost-lg pointer-events-none absolute inset-0 bg-[var(--pr-flag)]"
            />
            <blockquote
              data-reveal="block"
              className="pr-edge-2 relative overflow-hidden bg-[var(--pr-ink)] text-[var(--pr-stock)]"
            >
              <DotScreen id="pr-screen-note" tone="stock" />
              <div className="relative p-6 sm:p-10 lg:p-14">
                <Stamp tone="flag" className="mb-8">
                  Written first
                </Stamp>
                <p
                  className={clsx(
                    "max-w-[24ch] text-[length:var(--pr-lead)] leading-[0.98] font-bold tracking-[-0.035em] uppercase",
                  )}
                >
                  {content.evaluation.note}
                </p>
              </div>
            </blockquote>
          </div>
        </Section>

        {/* ── 05 Not shipping ─────────────────────────────────────────── */}
        <Section
          id="scope"
          number="05"
          slug="Not shipping"
          note={`${content.nonGoals.length} exclusions, declared`}
          title="What this refuses to be"
          deck={`${content.nonGoals.length} things this deliberately does not do, struck out here rather than discovered later. A scope this small is what makes the measurement mean anything — there is nothing else in the system to blame a bad number on.`}
          asideLabel="The trade"
          aside={
            <>
              Every exclusion buys the same thing: one corpus, one index, one language and no
              memory, so a result can only be caused by retrieval or by generation. Add a second
              tenant and the numbers stop being comparable.
            </>
          }
        >
          <ul data-group="skew" className="grid grid-cols-1 gap-5 sm:gap-6 md:grid-cols-2">
            {content.nonGoals.map((goal, index) => (
              <li key={goal} className="pr-edge flex items-stretch bg-[var(--pr-sheet)]">
                <span
                  aria-hidden
                  className="pr-edge flex w-12 shrink-0 items-center justify-center border-t-0 border-b-0 border-l-0 bg-[var(--pr-flag)] p-3 sm:w-14 sm:p-3.5"
                >
                  <StrikeMark />
                </span>
                <span className="flex min-w-0 flex-1 items-center gap-4 px-4 py-4 sm:px-5 sm:py-5">
                  <span
                    className={clsx(
                      MONO,
                      "shrink-0 text-[11px] font-bold tabular-nums text-[var(--pr-ink-3)]",
                    )}
                  >
                    5.{index + 1}
                  </span>
                  <span className="text-[14px] leading-[1.25] font-bold tracking-[-0.015em] uppercase sm:text-[15.5px]">
                    {goal}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </Section>
      </main>

      {/* ── Colophon ────────────────────────────────────────────────────
          Deep bottom padding: the review switcher is fixed to the bottom
          centre of the viewport, and a footer that runs under it is a footer
          nobody can read. */}
      <footer id="docs" className="scroll-mt-4">
        <div className="pr-edge-b-2 bg-[var(--pr-flag)] text-[var(--pr-on-flag)]">
          <div className={clsx(SHELL, "flex items-center gap-4 py-3")}>
            <RegMark className="h-4 w-4" />
            <span className={clsx(LABEL, "text-[var(--pr-on-flag)]")}>Colophon</span>
            <span aria-hidden className="ml-auto flex gap-2">
              <span className="block h-3.5 w-8 bg-[var(--pr-ink)]" />
              <span className="pr-edge block h-3.5 w-8 bg-[var(--pr-stock)]" />
              <span className="block h-3.5 w-8 bg-[var(--pr-proof)]" />
            </span>
          </div>
        </div>

        <div className={clsx(SHELL, "pt-14 pb-36 sm:pt-20 sm:pb-40")}>
          <div className="grid grid-cols-1 gap-x-8 gap-y-12 lg:grid-cols-12">
            <div className="lg:col-span-5">
              <p
                className={clsx(
                  "text-[2.5rem] leading-[0.85] font-bold tracking-[-0.055em] uppercase sm:text-[3.5rem]",
                )}
              >
                {content.brand.name}
              </p>
              <p
                className={clsx(
                  MONO,
                  "mt-4 text-[12px] tracking-[0.12em] text-[var(--pr-ink-2)] uppercase sm:text-[13px]",
                )}
              >
                {content.brand.tagline}
              </p>

              <a
                href={`https://${content.footer.repo}`}
                className={clsx(
                  MONO,
                  "pr-edge-2 pr-key mt-8 inline-flex items-center gap-3 bg-[var(--pr-sheet)] px-4 py-3 text-[12px] font-bold tracking-[0.04em] sm:px-5 sm:text-[13px]",
                )}
              >
                {content.footer.repo}
                <ArrowUpRight size={15} strokeWidth={3} aria-hidden />
              </a>
            </div>

            <div className="lg:col-span-3 lg:col-start-7">
              <h2 id="stack-label" className={clsx(LABEL, "text-[var(--pr-ink-3)]")}>
                Set in
              </h2>
              <ul aria-labelledby="stack-label" className="mt-4 flex flex-wrap gap-2">
                {content.footer.stack.map((item) => (
                  <li key={item}>
                    <Chip>{item}</Chip>
                  </li>
                ))}
              </ul>
            </div>

            <div className="lg:col-span-3 lg:col-start-10">
              <h2 className={clsx(LABEL, "text-[var(--pr-ink-3)]")}>Press notes</h2>
              <p className="pr-edge-t mt-4 pt-4 text-[12.5px] leading-[1.65] text-[var(--pr-ink-2)]">
                Display and text set in Space Grotesk, apparatus in JetBrains Mono. Two inks —
                black and a fluorescent orange-red — on warm stock, with a blueline third pass
                reserved for figures that have not been measured. Rules at two and three pixels.
                No gradients, no blur, one tilt.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

/**
 * The docket on the front page. Counts are derived from the content model so
 * the contents list cannot drift from the document underneath it.
 */
const SHEET_CONTENTS = [
  { id: "transcript", number: "01", slug: "The exchange", note: `${QUESTION_COUNT} questions` },
  { id: "pipeline", number: "02", slug: "The run", note: `${STAGE_COUNT} stages` },
  { id: "results", number: "03", slug: "Results", note: `${MEASURE_COUNT} measures` },
  {
    id: "evaluation",
    number: "04",
    slug: "The set",
    note: `${content.evaluation.points.length} strata`,
  },
  { id: "scope", number: "05", slug: "Not shipping", note: `${content.nonGoals.length} excluded` },
];

"use client";

import { BotAvatar } from "bot-avatars";
import { cn } from "@/lib/utils";

/**
 * YOUR LOGO SLOT.
 * Default: the 3D plastic ghost from `bot-avatars`.
 * To use your own agent icon, drop an image (svg/png/webp) in `web/public/` and set
 *     NEXT_PUBLIC_QUANTA_LOGO=/your-icon.svg
 * in `web/.env.local` (restart `npm run dev`). Everything that shows the agent
 * (hero, messages, top bar) switches automatically.
 */
const CUSTOM_LOGO_SRC = process.env.NEXT_PUBLIC_QUANTA_LOGO ?? "";

export interface QuantaLogoProps {
  size?: number;
  /** Quanta is generating an answer: the ghost hops and smiles. */
  working?: boolean;
  /** Freeze the animation (cheaper for long message lists). */
  paused?: boolean;
  /** Eyes follow the pointer and a click makes it hop (avatar only). */
  interactive?: boolean;
  className?: string;
}

export function QuantaLogo({ size = 40, working = false, paused = false, interactive = true, className }: QuantaLogoProps) {
  if (CUSTOM_LOGO_SRC) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={CUSTOM_LOGO_SRC}
        alt="Quanta"
        width={size}
        height={size}
        className={cn("shrink-0 select-none object-contain", working && "animate-pulse", className)}
        draggable={false}
      />
    );
  }

  return (
    <BotAvatar
      type="ghost"
      shading="plastic"
      theme="dark"
      size={size}
      state={working ? "working" : "default"}
      paused={paused}
      interactive={interactive}
      aria-label={working ? "Quanta is working" : "Quanta"}
      className={cn("shrink-0", className)}
    />
  );
}

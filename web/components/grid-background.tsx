 export function GridBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 -z-10 flex h-full w-full items-center justify-center">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:32px_32px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
    </div>
  );
}
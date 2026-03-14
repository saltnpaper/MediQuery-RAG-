import * as React from "react";

import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "flex h-11 w-full rounded-md border px-3 py-2 text-sm text-white outline-none placeholder:text-gray-500 focus:border-green-500",
        className,
      )}
      {...props}
    />
  ),
);

Input.displayName = "Input";

// 界面上显示的本地时间。
const pad = number => String(number).padStart(2, '0');

/** 本地时间 `MM-DD HH:mm`。 */
export function formatTime(ms) {
  const date = new Date(ms);
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${formatClock(ms)}`;
}

/** 本地时间 `HH:mm`。 */
export function formatClock(ms) {
  const date = new Date(ms);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

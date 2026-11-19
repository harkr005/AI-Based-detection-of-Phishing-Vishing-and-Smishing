import { GoogleGenAI, Type } from "@google/genai";
import { userId } from "@/lib/auth";

const VERBATIM =
  "Transcribe verbatim: preserve URLs, phone numbers, sender IDs, amounts and " +
  "spelling mistakes character for character — they are the evidence being " +
  "analysed, so never correct, translate, summarise or reformat them. Leave a " +
  "field as an empty string if it is not visible in the screenshot.";

const str = (description) => ({ type: Type.STRING, description });

// The only place a real SEND time can come from in this app. The behaviour
// layer's odd-hour rule abstains without one, because the clock when someone
// pastes a message is not the clock when it was delivered — a message read at
// 2am was not sent at 2am. Screenshots do carry it, so read it here or not at all.
const SENT_AT = str(
  "The time the message itself shows it was SENT — the bubble timestamp, the " +
  "date separator above it, or the mail header's date. Return it as an " +
  "ISO-8601 datetime. If only a clock time is visible (\"10:42 pm\"), resolve " +
  "it against the current date given in the instruction, taking the most recent " +
  "occurrence at or before that moment. Empty string if no time is visible — " +
  "never guess one."
);

/** One shape per channel so the extracted parts land in the right form field. */
const SPEC = {
  sms: {
    schema: {
      type: Type.OBJECT,
      properties: {
        sender: str(
          "The sender ID / header of the conversation, taken from the title bar " +
          "at the top of the screen (e.g. \"VM-ECIsms-G\", \"AX-HDFCBK\", or a " +
          "phone number). Never a name from inside the message text."
        ),
        text: str(
          "Only the single most recent message from that sender — the bottom-most " +
          "message bubble. Ignore every older message above it, date separators, " +
          "timestamps, \"Unread\" dividers, the phone status bar, and carrier or " +
          "app notices such as \"Sender can't accept replies\" or \"Learn more\"."
        ),
        sent_at: SENT_AT,
      },
      required: ["sender", "text"],
    },
    prompt: "This is a screenshot of an SMS conversation.",
  },
  email: {
    schema: {
      type: Type.OBJECT,
      properties: {
        sender: str(
          "The From line: display name plus the email address in angle brackets " +
          "if both are shown, e.g. \"Example Store Offers <offers@example-store.in>\"."
        ),
        subject: str("The subject line, without labels like \"Inbox\" or \"External\"."),
        body: str(
          "All visible body text of the email, in reading order, including text " +
          "rendered inside banner images, headings, prices, button labels and the " +
          "footer. Ignore the mail client's own chrome: toolbars, sidebars, the " +
          "reply/forward buttons, and the \"1 of 390\" counter."
        ),
        sent_at: SENT_AT,
      },
      required: ["sender", "subject", "body"],
    },
    prompt: "This is a screenshot of an email opened in a mail client.",
  },
};

export async function POST(req) {
  if (!(await userId())) return Response.json({ error: "Unauthorised" }, { status: 401 });
  if (!process.env.GEMINI_API_KEY)
    return Response.json({ error: "GEMINI_API_KEY is not set" }, { status: 500 });

  const form = await req.formData();
  const file = form.get("image");
  const spec = SPEC[form.get("channel")] ?? SPEC.sms;
  if (!file) return Response.json({ error: "No image uploaded" }, { status: 400 });

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  try {
    const res = await ai.models.generateContent({
      model: "gemini-3.6-flash",
      contents: [{
        role: "user",
        parts: [
          { inlineData: {
              mimeType: file.type || "image/png",
              data: Buffer.from(await file.arrayBuffer()).toString("base64"),
          } },
          { text: `${spec.prompt} Fill each field of the schema from it. ${VERBATIM} ` +
                  `The current date and time is ${new Date().toString()}.` },
        ],
      }],
      config: { responseMimeType: "application/json", responseSchema: spec.schema },
    });
    return Response.json(JSON.parse(res.text));
  } catch (e) {
    return Response.json({ error: e.message }, { status: 502 });
  }
}
